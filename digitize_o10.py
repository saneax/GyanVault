import sqlite3
import os
import shutil
import gc
import json
import torch
from pathlib import Path
from pdf2image import convert_from_path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

# --- CONFIGURATION ---
DB_PATH = "downloads.db"
ROOT_OUTPUT_DIR = "./output"  # Root of your downloaded PDFs
STAGING_DIR = "./staging"     # Temporary area for images/raw text
FINAL_DIR = "./digitized"     # Where final JSONs go

# Create directories
os.makedirs(STAGING_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# =========================================================================
# PHASE 1: DISCOVERY & VISION (The "Eye")
# =========================================================================
def run_pass_1_vision():
    print("\n" + "="*50)
    print(">>> PASS 1: VISION & TRANSCRIPTION")
    print("="*50)

    # 1. Connect to DB to get the list of files we EXPECT to find
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Assuming table is 'downloads' or similar. Adjust query to your actual schema.
    # We fetch relevant columns to verify file existence.
    try:
        cursor.execute("SELECT id, year, class, subject, filepath FROM downloads WHERE processed_json IS NULL")
        records = cursor.fetchall()
    except sqlite3.OperationalError:
        print("!! DB Schema Error: Could not query 'downloads' table. Please check table/column names.")
        return

    if not records:
        print(">>> No pending files found in DB (or all marked processed).")
        return

    # 2. Load Vision Model (Qwen2-VL-2B) - Optimized for 4060
    print(">>> Loading Vision Model...")
    model_path = "Qwen/Qwen2-VL-2B-Instruct"
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="flash_attention_2"
        )
    except:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
        )
    processor = AutoProcessor.from_pretrained(model_path)

    # 3. Process each file
    for row in records:
        file_id = row['id']
        rel_path = row['filepath'] # e.g., "2023/XII/Download/BusinessStudies.pdf"
        full_pdf_path = os.path.join(ROOT_OUTPUT_DIR, rel_path)
        
        # Define output text path (using ID to keep it flat and safe in staging)
        raw_text_path = os.path.join(STAGING_DIR, f"{file_id}_raw.txt")
        
        if os.path.exists(raw_text_path):
            print(f"-> Skipping {rel_path} (Raw text exists)")
            continue

        if not os.path.exists(full_pdf_path):
            print(f"!! File not found on disk: {full_pdf_path}")
            continue

        print(f"-> Processing: {rel_path}")

        # A. Convert PDF to Images
        try:
            images = convert_from_path(full_pdf_path, dpi=300)
        except Exception as e:
            print(f"   !! PDF Error: {e}")
            continue

        full_text_content = ""

        # B. OCR Each Page
        for i, image in enumerate(images):
            # Save temp image for VLM
            temp_img_path = os.path.join(STAGING_DIR, "temp_vision.png")
            image.save(temp_img_path)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": temp_img_path},
                    {"type": "text", "text": "Transcribe this page line by line. Do not interpret. Just text."}
                ]
            }]
            
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            ).to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            page_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            
            full_text_content += f"\n--- PAGE {i+1} ---\n{page_text}\n"

        # C. Save Raw Text
        with open(raw_text_path, "w", encoding="utf-8") as f:
            f.write(full_text_content)
            print(f"   [Saved Raw Text ID: {file_id}]")

    # Cleanup
    del model
    del processor
    cleanup_gpu()


# =========================================================================
# PHASE 2: LOGIC & CONTEXT (The "Brain")
# =========================================================================
def run_pass_2_logic():
    print("\n" + "="*50)
    print(">>> PASS 2: LOGIC & JSON FORMATTING")
    print("="*50)

    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Identify files that have Raw Text but no Final JSON
    # (We scan the staging dir for _raw.txt files)
    pending_files = [f for f in os.listdir(STAGING_DIR) if f.endswith("_raw.txt")]
    
    if not pending_files:
        print(">>> No raw text files waiting for JSON conversion.")
        return

    # 2. Load Logic Model (Qwen2.5-7B-Int4)
    print(">>> Loading Logic Model...")
    model_id = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype="auto")

    for filename in pending_files:
        file_id = filename.split("_")[0] # Extract ID from "55_raw.txt"
        
        # A. Fetch Context from DB
        cursor.execute("SELECT * FROM downloads WHERE id = ?", (file_id,))
        record = cursor.fetchone()
        
        if not record:
            print(f"!! Orphaned text file: {filename} (No DB record)")
            continue

        # B. Construct the Contextual Prompt
        # We INJECT the DB knowledge into the prompt so the LLM doesn't have to guess.
        metadata_context = {
            "subject": record['subject'],
            "class": record['class'],
            "year": record['year'],
            "source_file": record['filepath']
        }
        
        print(f"-> Structuring JSON for ID {file_id} ({metadata_context['subject']})")

        with open(os.path.join(STAGING_DIR, filename), "r", encoding="utf-8") as f:
            raw_text = f.read()

        prompt = f"""
        You are a strict data extraction engine.
        
        CONTEXT FROM DATABASE:
        Subject: {metadata_context['subject']}
        Class: {metadata_context['class']}
        Year: {metadata_context['year']}
        
        TASK:
        Convert the provided raw OCR text into valid JSON.
        1. Use the "CONTEXT" values above to fill the metadata fields.
        2. Extract questions into a list.
        3. Preserve Hindi text exactly as is.
        
        RAW TEXT:
        {raw_text[:3500]} 
        
        REQUIRED JSON STRUCTURE:
        {{
            "metadata": {{
                "subject": "{metadata_context['subject']}",
                "class": "{metadata_context['class']}",
                "year": "{metadata_context['year']}"
            }},
            "questions": [
                {{ "q_no": "1", "text": "Question text here...", "marks": "5" }}
            ]
        }}
        
        OUTPUT JSON ONLY:
        """

        messages = [
            {"role": "system", "content": "You are a JSON generator. Do not speak. Only output JSON."},
            {"role": "user", "content": prompt}
        ]

        # C. Inference
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        
        outputs = model.generate(inputs.input_ids, max_new_tokens=2048, temperature=0.1)
        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
        json_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # D. Post-Process & Save
        # Try to parse it to ensure validity, or save raw if it fails
        final_json_path = os.path.join(FINAL_DIR, f"{file_id}_{metadata_context['subject']}.json")
        
        try:
            # Simple cleanup to remove markdown code blocks if the model adds them
            clean_json = json_str.replace("```json", "").replace("```", "").strip()
            parsed_json = json.loads(clean_json)
            
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            
            # Optional: Update DB to mark as done
            # cursor.execute("UPDATE downloads SET processed_json = ? WHERE id = ?", (final_json_path, file_id))
            # conn.commit()
            print(f"   [Success] Saved to {final_json_path}")
            
        except json.JSONDecodeError:
            print(f"   [Warning] Model produced invalid JSON. Saving raw output.")
            with open(final_json_path.replace(".json", ".err"), "w", encoding="utf-8") as f:
                f.write(json_str)

    cleanup_gpu()

if __name__ == "__main__":
    # Ensure dependencies are installed:
    # pip install opencv-python-headless pdf2image
    
    # 1. Run Vision (OCR)
    run_pass_1_vision()
    
    # 2. Run Logic (JSON)
    run_pass_2_logic()


