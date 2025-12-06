import sqlite3
import os
import json
import gc
import torch
import hashlib
import re
import argparse
from pathlib import Path
from pdf2image import convert_from_path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image

# --- CONFIGURATION ---
DB_PATH = "downloads.db"
ROOT_OUTPUT_DIR = "output"
DEBUG_STAGING_DIR = "./debug_staging"
DEBUG_FINAL_DIR = "./debug_final"

os.makedirs(DEBUG_STAGING_DIR, exist_ok=True)
os.makedirs(DEBUG_FINAL_DIR, exist_ok=True)

# --- VRAM SAFETY SETTINGS ---
OCR_DPI = 150
MAX_IMAGE_DIMENSION = 1024

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def get_unique_file_id(file_path):
    return hashlib.md5(file_path.encode()).hexdigest()[:10]

def find_random_pdfs(subject, num_pdfs):
    """Finds a random selection of PDF files for a given subject from the database."""
    print(f"Searching for {num_pdfs} random PDFs for subject: '{subject}'")
    conn = get_db_connection()
    cursor = conn.cursor()

    # This query is more robust. It searches for the subject in the 'subject' column,
    # the 'path' column, and the 'pdfs_json' column. This accounts for cases where
    # the subject metadata might be in the file path rather than the subject field.
    search_term = f"%{subject.lower()}%"
    query = """
        SELECT complete_url, year, class, subject, path, pdfs_json
        FROM downloads
        WHERE (LOWER(subject) LIKE ? OR LOWER(path) LIKE ? OR LOWER(pdfs_json) LIKE ?)
          AND (LOWER(path) LIKE '%.pdf' OR pdfs_json LIKE '%pdf%')
        ORDER BY RANDOM()
        LIMIT ?
    """
    try:
        cursor.execute(query, (search_term, search_term, search_term, num_pdfs))
        records = cursor.fetchall()
        conn.close()
        print(f"Found {len(records)} matching records in the database.")
        return records
    except sqlite3.OperationalError as e:
        print(f"!! DB Error: {e}")
        conn.close()
        return []

def run_pass_1_vision(pdf_path, file_id):
    """Runs the vision model to perform OCR on a single PDF."""
    print("\n" + "="*30)
    print(f">>> PASS 1: VISION for {pdf_path.name}")
    print("="*30)

    # 1. Load Model
    model_path = "Qwen/Qwen2-VL-7B-Instruct"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto", attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(model_path)

    raw_text_path = os.path.join(DEBUG_STAGING_DIR, f"{file_id}_raw.txt")
    full_text_content = ""

    try:
        # 2. Convert and Process PDF
        images = convert_from_path(pdf_path, dpi=OCR_DPI)
        for i, image in enumerate(images):
            print(f"  - Processing page {i+1}/{len(images)}")
            if max(image.size) > MAX_IMAGE_DIMENSION:
                image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)

            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Transcribe this page line-by-line. Preserve LaTeX and Hindi text."}
            ]}]

            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text_prompt], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=1024)

            generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            page_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            full_text_content += f"\n--- PAGE {i+1} ---\n{page_text}\n"
            del inputs, generated_ids
            cleanup_gpu()

        with open(raw_text_path, "w", encoding="utf-8") as f:
            f.write(full_text_content)
        print(f"  [Success] Raw text saved to {raw_text_path}")

    except Exception as e:
        print(f"  !! Error in Pass 1: {e}")
        return None
    finally:
        # 3. Unload Model
        del model, processor
        cleanup_gpu()
        print("  Pass 1 Model Unloaded.")

    return raw_text_path

def preprocess_raw_text(raw_text):
    """
    Preprocesses the raw OCR text to isolate and clean the relevant question sections.
    1. Finds the start of the actual questions (e.g., "SECTION A").
    2. Filters out all non-ASCII (non-English) characters.
    """
    # Find the starting point of the questions
    start_marker_eng = "SECTION A"
    start_marker_hin = "खण्ड क"
    start_pos_eng = raw_text.find(start_marker_eng)
    start_pos_hin = raw_text.find(start_marker_hin)

    start_pos = min(p for p in [start_pos_eng, start_pos_hin] if p != -1) if any(p != -1 for p in [start_pos_eng, start_pos_hin]) else 0
    
    # Keep only ASCII characters (i.e., English and standard symbols)
    return "".join(char for char in raw_text[start_pos:] if ord(char) < 128)

def run_pass_2_logic(raw_text_path, file_id, metadata):
    """Runs the logic model to convert raw text to structured JSON."""
    print("\n" + "="*30)
    print(f">>> PASS 2: LOGIC for {file_id}")
    print("="*30)

    # 1. Load Model
    model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", attn_implementation="sdpa")

    final_path = os.path.join(DEBUG_FINAL_DIR, f"{file_id}_{metadata.get('subject', 'unknown')}.json")
    error_path = final_path + ".err"

    try:
        with open(raw_text_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # NEW: Preprocess the text to make it easier for the model
        print("  - Preprocessing text (keeping English only, finding question start)...")
        clean_text = preprocess_raw_text(raw_text)

        # 2. Generate JSON
        # REINFORCED PROMPT: More explicit instructions and a detailed example.
        prompt = rf"""
<|im_start|>system
You are an expert data extraction engine for academic papers. Your task is to convert the provided text into a single, valid JSON object.

CRITICAL RULES:
1. You MUST output a single, valid JSON object and nothing else. Do not use markdown.
2. You MUST escape all backslashes for JSON compatibility. For example, `\sin` becomes `"\\sin"`.
3. Extract questions, their options (for MCQs), and marks.
4. For the "explanation" field, provide a detailed, step-by-step walkthrough of how to arrive at the correct answer. Behave like a teacher explaining the concept.
5. Ignore any leftover instructional text or page markers. Focus only on the questions.

EXAMPLE JSON STRUCTURE:
{{
  "questions": [
    {{
      "q_no": "1",
      "text": "If \(100 \\equiv x(\\bmod 7)\\), then the least positive value of \(x\) is :",
      "options": ["(a) 6", "(b) 4", "(c) 3", "(d) 2"],
      "marks": "1",
      "answer": "The correct option is (d) 2.",
      "explanation": "The expression \\(100 \\equiv x(\\bmod 7)\\) means that 100 and x have the same remainder when divided by 7. First, we divide 100 by 7. 100 divided by 7 is 14 with a remainder of 2 (since 14 * 7 = 98). Therefore, the remainder is 2. The question asks for the least positive value of x, which is this remainder. So, x = 2."
    }}
  ]
}}
<|im_end|>
<|im_start|>user
CONTEXT:
- Subject: {metadata.get('subject', 'N/A')}
- Class: {metadata.get('class', 'N/A')}
- Year: {metadata.get('year', 'N/A')}

CLEANED ENGLISH TEXT TO CONVERT:
{clean_text[:4000]}
<|im_end|>
<|im_start|>assistant
        """
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=3072, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
        
        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
        json_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 3. Clean and Save
        try:
            clean_json_str = json_str[json_str.find('{') : json_str.rfind('}')+1]
            # This regex is more robust. It finds any single backslash that is not
            # already escaped and escapes it. This is a critical step to fix model
            # errors before parsing. It handles cases like `\n`, `\t`, `\(` and `\s` etc.
            # It looks for a `\` that is not followed by another `\` (negative lookahead).
            corrected_json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', clean_json_str)

            parsed_json = json.loads(corrected_json_str)
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            print(f"  [Success] Final JSON saved to {final_path}")
        except json.JSONDecodeError as e:
            print(f"  [Fail] JSON was invalid: {e}. Saving raw output to {error_path}")
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(json_str)

    except Exception as e:
        print(f"  !! Error in Pass 2: {e}")
    finally:
        # 4. Unload Model
        del model, tokenizer
        cleanup_gpu()
        print("  Pass 2 Model Unloaded.")

def get_metadata_for_id(file_id):
    """Fetches metadata from the database for a given file_id by reverse-searching the path."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # This is a bit of a brute-force search, but necessary since we only have the ID.
    cursor.execute("SELECT complete_url, year, class, subject, path, pdfs_json FROM downloads")
    records = cursor.fetchall()
    conn.close()

    for row in records:
        # Re-generate the ID from the path to find the matching record
        rel_path = None
        if row['pdfs_json'] and row['pdfs_json'] != "[]":
            try:
                pdf_list = json.loads(row['pdfs_json'])
                if pdf_list: rel_path = pdf_list[0]['file']
            except json.JSONDecodeError: pass
        
        if not rel_path and row['path'] and row['path'].lower().endswith('.pdf'):
            rel_path = str(Path(row['path']).relative_to(ROOT_OUTPUT_DIR))

        if rel_path and get_unique_file_id(str(rel_path)) == file_id:
            return dict(row)
            
    return None

def main():
    # MODIFIED: Skip Pass 1 and directly scan the debug_staging directory.
    print("Scanning for existing raw text files in ./debug_staging/...")
    
    staged_files = [f for f in os.listdir(DEBUG_STAGING_DIR) if f.endswith("_raw.txt")]

    if not staged_files:
        print("No '_raw.txt' files found in ./debug_staging/.")
        return

    print(f"Found {len(staged_files)} files to process for Pass 2.")

    for txt_filename in staged_files:
        file_id = txt_filename.split("_raw.txt")[0]
        raw_text_file = os.path.join(DEBUG_STAGING_DIR, txt_filename)
        
        print(f"\n{'='*50}\nProcessing file ID: {file_id}\n{'='*50}")
        
        metadata = get_metadata_for_id(file_id)

        if not metadata:
            print(f"  !! Error: Could not find metadata in the database for file ID '{file_id}'. Skipping.")
            continue

        if raw_text_file:
            run_pass_2_logic(raw_text_file, file_id, metadata)

if __name__ == "__main__":
    main()