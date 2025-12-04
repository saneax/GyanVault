import sqlite3
import os
import json
import gc
import torch
import hashlib
from pathlib import Path
from pdf2image import convert_from_path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

# --- CONFIGURATION ---
DB_PATH = "downloads.db"
ROOT_OUTPUT_DIR = "output"    # Folder where crawler saved files
STAGING_DIR = "./staging"     # Temp storage for OCR text
FINAL_DIR = "./digitized"     # Final JSON output

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

def get_unique_file_id(file_path):
    """Generate a stable short ID from the file path to name output files."""
    return hashlib.md5(file_path.encode()).hexdigest()[:10]

# =========================================================================
# PHASE 1: DISCOVERY & VISION (The "Eye")
# =========================================================================
def run_pass_1_vision():
    print("\n" + "="*50)
    print(">>> PASS 1: VISION & TRANSCRIPTION")
    print("="*50)

    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Select columns that ACTUALLY exist in your crawler DB
    try:
        cursor.execute("SELECT complete_url, year, class, subject, path, pdfs_json FROM downloads")
        records = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"!! DB Error: {e}")
        return

    print(">>> Loading Vision Model...")
    model_path = "Qwen/Qwen2-VL-2B-Instruct"
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="flash_attention_2"
        )
    except:
        print(">> Flash Attention not found, using SDPA.")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
        )
    processor = AutoProcessor.from_pretrained(model_path)

    for row in records:
        # 1. Determine the actual PDF files to process
        target_files = []
        
        # Check if it was a zip with extracted PDFs
        pdfs_json_str = row['pdfs_json']
        main_path = row['path']
        
        if pdfs_json_str and pdfs_json_str != "[]":
            try:
                pdf_list = json.loads(pdfs_json_str)
                # 'file' in json is relative path like "2023/XII/Subject/paper.pdf"
                target_files = [p['file'] for p in pdf_list]
            except json.JSONDecodeError:
                target_files = []
        
        # If no PDFs in JSON, check if the main path is a PDF
        if not target_files and main_path and main_path.lower().endswith('.pdf'):
            # The crawler stores full path in 'path' sometimes, or relative?
            # Looking at crawler script, 'path' seems to include 'output/' prefix or is absolute?
            # Let's handle both.
            if str(main_path).startswith("output/"):
                 target_files = [str(main_path).replace("output/", "", 1)]
            else:
                 target_files = [main_path]

        if not target_files:
            continue

        # 2. Process each PDF found for this DB entry
        for rel_path in target_files:
            # Construct full path. Crawler saves in output/...
            full_pdf_path = os.path.join(ROOT_OUTPUT_DIR, rel_path)
            
            # Generate ID for staging files
            file_id = get_unique_file_id(rel_path)
            raw_text_path = os.path.join(STAGING_DIR, f"{file_id}_raw.txt")
            
            # Check if done
            if os.path.exists(raw_text_path):
                print(f"-> Skipping {rel_path} (Raw text exists)")
                continue
                
            if not os.path.exists(full_pdf_path):
                # Try prepending output/ if missing
                full_pdf_path = os.path.join(ROOT_OUTPUT_DIR, "output", rel_path) 
                if not os.path.exists(full_pdf_path):
                    # print(f"!! File not found: {full_pdf_path}")
                    continue

            print(f"-> Processing: {rel_path}")

            try:
                images = convert_from_path(full_pdf_path, dpi=300)
            except Exception as e:
                print(f"   !! PDF Error: {e}")
                continue

            full_text_content = ""

            for i, image in enumerate(images):
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

            # Save Raw Text AND Metadata map
            with open(raw_text_path, "w", encoding="utf-8") as f:
                f.write(full_text_content)
            
            # Save a tiny metadata sidecar so Pass 2 knows the context
            meta = {
                "subject": row['subject'],
                "class": row['class'],
                "year": row['year'],
                "original_path": rel_path
            }
            with open(os.path.join(STAGING_DIR, f"{file_id}_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)

    del model
    del processor
    cleanup_gpu()


# =========================================================================
# PHASE 2: LOGIC & JSON FORMATTING (The "Brain")
# =========================================================================
def run_pass_2_logic():
    print("\n" + "="*50)
    print(">>> PASS 2: LOGIC & JSON FORMATTING")
    print("="*50)

    # Scan staging for files that have text but no final JSON
    pending_files = []
    for f in os.listdir(STAGING_DIR):
        if f.endswith("_raw.txt"):
            file_id = f.split("_")[0]
            # Check if final exists
            # We need to know the subject to name it strictly, or just use ID
            # Let's load the meta file to check name
            meta_path = os.path.join(STAGING_DIR, f"{file_id}_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as mf:
                    meta = json.load(mf)
                final_name = f"{file_id}_{meta.get('subject', 'unknown')}.json"
                if not os.path.exists(os.path.join(FINAL_DIR, final_name)):
                    pending_files.append((f, meta))

    if not pending_files:
        print(">>> No pending files.")
        return

    print(">>> Loading Logic Model...")
    model_id = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype="auto")

    for filename, meta in pending_files:
        file_id = filename.split("_")[0]
        print(f"-> Structuring JSON for {meta.get('subject')}")

        with open(os.path.join(STAGING_DIR, filename), "r", encoding="utf-8") as f:
            raw_text = f.read()

        prompt = f"""
        You are a data extractor.
        
        CONTEXT:
        Subject: {meta['subject']}
        Class: {meta['class']}
        Year: {meta['year']}
        
        TASK:
        Convert the OCR text into JSON.
        1. Extract questions list.
        2. Preserve Hindi text.
        
        RAW TEXT:
        {raw_text[:3500]} 
        
        OUTPUT JSON Format:
        {{
            "metadata": {{ "subject": "{meta['subject']}", "year": "{meta['year']}" }},
            "questions": [ {{ "q_no": "1", "text": "...", "marks": "..." }} ]
        }}
        """

        messages = [
            {"role": "system", "content": "Output valid JSON only."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        
        outputs = model.generate(inputs.input_ids, max_new_tokens=2048, temperature=0.1)
        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
        json_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        final_name = f"{file_id}_{meta.get('subject', 'unknown')}.json"
        final_path = os.path.join(FINAL_DIR, final_name)
        
        try:
            clean_json = json_str.replace("```json", "").replace("```", "").strip()
            parsed_json = json.loads(clean_json)
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            print(f"   [Success] Saved {final_name}")
        except:
            print(f"   [Fail] JSON invalid. Saving raw.")
            with open(final_path + ".err", "w", encoding="utf-8") as f:
                f.write(json_str)

    cleanup_gpu()

if __name__ == "__main__":
    run_pass_1_vision()
    run_pass_2_logic()
