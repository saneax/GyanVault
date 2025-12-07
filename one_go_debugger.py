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

def find_staged_files(num_files):
    """Finds random raw text files in the staging directory and gets their metadata."""
    print(f"Searching for {num_files} random processed files in: '{DEBUG_STAGING_DIR}'")
    staged_files = [f for f in os.listdir(DEBUG_STAGING_DIR) if f.endswith("_raw.txt")]

    if not staged_files:
        return []

    conn = get_db_connection()
    cursor = conn.cursor()
    # This is inefficient, but for a debug script it's acceptable.
    # It iterates through the DB to find records matching the staged files.
    cursor.execute("SELECT complete_url, year, class, subject, path, pdfs_json FROM downloads")
    all_records = cursor.fetchall()
    conn.close()

    found_records = []
    for record in all_records:
        rel_path = get_rel_path_from_record(record)
        if rel_path:
            file_id = get_unique_file_id(str(rel_path))
            if f"{file_id}_raw.txt" in staged_files:
                found_records.append(record)

    return found_records[:num_files] # Return the requested number of files

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
                {"type": "text", "text": "Transcribe this page line-by-line. IMPORTANT: Only transcribe the English text. Ignore all Hindi text."}
            ]}]

            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text_prompt], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=1024)

            generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            page_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

            # AGGRESSIVE FILTER: Remove all non-ASCII characters to guarantee English-only text.
            page_text_english_only = "".join(char for char in page_text if ord(char) < 128)
            full_text_content += f"\n--- PAGE {i+1} ---\n{page_text_english_only}\n"
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

def parse_and_fix_json(json_str: str) -> dict | None:
    """
    A more robust JSON parser that attempts to fix common LLM-generated errors,
    especially around unescaped backslashes.
    """
    # 1. Find the main JSON object
    start_index = json_str.find('{')
    end_index = json_str.rfind('}')
    if start_index == -1 or end_index == -1:
        return None
    json_str = json_str[start_index : end_index + 1]

    # 2. Iteratively try to fix and parse
    for i in range(5): # Try to fix up to 5 times
        try:
            # Attempt to remove trailing commas before closing brackets/braces
            clean_str = re.sub(r",\s*([}\]])", r"\1", json_str)
            return json.loads(clean_str)
        except json.JSONDecodeError as e:
            # If it's an escape error, find the problematic character and escape it
            if "Invalid \\escape" in str(e):
                # The error message gives us the position of the bad character
                bad_char_pos = e.pos + (json_str.find('{') if json_str.find('{') > -1 else 0)
                # Insert an extra backslash right before the problematic character
                json_str = json_str[:bad_char_pos] + '\\' + json_str[bad_char_pos:]
            else: # For other errors, we can't reliably fix them
                raise e # Re-raise the exception to be caught outside
    return None

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

    all_questions = []

    try:
        # CHUNKING IMPLEMENTED: Process the document page by page.
        system_prompt = """
<|im_start|>system
You are an expert teacher and data extractor for academic papers. Your task is to convert the provided text into a single, valid JSON object.

CRITICAL RULES:
1.  You MUST output a single, valid JSON object and nothing else. Do not use markdown.
2.  You MUST escape all backslashes for JSON compatibility. For example, `\sin` must be written as `"\\sin"`.
3.  Extract questions, their options (for MCQs), and marks.
4.  For the "answer" field, provide a detailed, step-by-step walkthrough of how to arrive at the correct answer. Behave like a teacher explaining the concept.
5.  Ignore any leftover instructional text or page markers. Focus only on the questions.
6.  The output must be a JSON object containing a single key "questions", which is a list of question objects.

EXAMPLE JSON STRUCTURE:
{{
  "questions": [
    {{
      "q_no": "1",
      "text": "If \\(100 \\equiv x(\\bmod 7)\\), then the least positive value of \\(x\\) is:",
      "options": ["(a) 6", "(b) 4", "(c) 3", "(d) 2"],
      "marks": "1",
      "answer": "The correct option is (d) 2. The expression \\(100 \\equiv x(\\bmod 7)\\) means that 100 and x have the same remainder when divided by 7. First, we divide 100 by 7. 100 divided by 7 is 14 with a remainder of 2 (since 14 * 7 = 98). Therefore, the remainder is 2. The question asks for the least positive value of x, which is this remainder. So, x = 2."
    }}
  ]
}}
<|im_end|>"""

        with open(raw_text_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Split the document into pages based on our marker from Pass 1
        pages = raw_text.split("--- PAGE")[1:]

        for i, page_content in enumerate(pages):
            print(f"  - Pass 2: Processing Page {i+1}/{len(pages)}")
            
            user_prompt = f"""
<|im_start|>user
CONTEXT:
- Subject: {metadata.get('subject', 'N/A')}
- Class: {metadata.get('class', 'N/A')}
- Year: {metadata.get('year', 'N/A')}

PAGE TEXT TO CONVERT:
```text
{page_content}
```<|im_end|>
<|im_start|>assistant
"""
            inputs = tokenizer([system_prompt + user_prompt], return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=3072, temperature=0.05, do_sample=True, pad_token_id=tokenizer.eos_token_id)

            generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
            json_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            try:
                parsed_json = parse_and_fix_json(json_str)
                if not parsed_json: continue

                if "questions" in parsed_json and isinstance(parsed_json["questions"], list):
                    all_questions.extend(parsed_json["questions"])

            except json.JSONDecodeError as e:
                print(f"    [Warn] JSON was invalid on page {i+1}: {e}. Saving raw output to {error_path}.page{i+1}")
                with open(f"{error_path}.page{i+1}", "w", encoding="utf-8") as f_err:
                    f_err.write(json_str)
            finally:
                del inputs, outputs, generated_ids
                cleanup_gpu()

        # After processing all pages, assemble the final document
        if all_questions:
            final_document = {"questions": all_questions}
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(final_document, f, indent=2, ensure_ascii=False)
            print(f"  [Success] Final JSON saved to {final_path} with {len(all_questions)} questions.")
        else:
            print(f"  [Fail] No questions could be extracted from any page.")

    except Exception as e:
        print(f"  !! Error in Pass 2: {e}")
    finally:
        # 4. Unload Model
        del model, tokenizer
        cleanup_gpu()
        print("  Pass 2 Model Unloaded.")

def get_rel_path_from_record(record_row):
    """Extracts the relative file path from a database row."""
    rel_path = None
    if record_row['pdfs_json'] and record_row['pdfs_json'] != "[]":
        try:
            pdf_list = json.loads(record_row['pdfs_json'])
            if pdf_list:
                rel_path = pdf_list[0]['file'] # Just take the first PDF from a zip
        except json.JSONDecodeError:
            pass
    
    if not rel_path and record_row['path'] and record_row['path'].lower().endswith('.pdf'):
        rel_path = str(Path(record_row['path']).relative_to(ROOT_OUTPUT_DIR))
    
    return rel_path

def process_file(row, run_pass_mode):
    """Processes a single file based on the selected pass mode."""
    rel_path = get_rel_path_from_record(row)
    if not rel_path:
        print(f"Could not determine a PDF path for record URL: {row['complete_url']}. Skipping.")
        return

    if run_pass_mode != '2': # For pass 1 or all, we need the actual PDF
        full_pdf_path = Path(ROOT_OUTPUT_DIR) / rel_path
        if not full_pdf_path.exists():
            print(f"File not found on disk: {full_pdf_path}. Skipping.")
            return

        file_id = get_unique_file_id(str(rel_path))
        metadata = dict(row)
        print(f"\n{'='*50}\nProcessing file: {full_pdf_path}\n{'='*50}")
    else: # For pass 2 only, we don't need the PDF path, just the ID and metadata
        file_id = get_unique_file_id(str(rel_path))
        metadata = dict(row)
        print(f"\n{'='*50}\nProcessing file ID: {file_id}\n{'='*50}")

    # --- Run the selected pipeline pass(es) ---
    raw_text_file = os.path.join(DEBUG_STAGING_DIR, f"{file_id}_raw.txt")

    if run_pass_mode in ["1", "all"]:
        # Run Pass 1, which generates the raw_text_file
        generated_raw_path = run_pass_1_vision(full_pdf_path, file_id)
        if not generated_raw_path:
            print(f"  [Fail] Pass 1 failed for {full_pdf_path}. Skipping Pass 2.")
            return

    if run_pass_mode in ["2", "all"]:
        # Run Pass 2, which requires the raw_text_file to exist
        if os.path.exists(raw_text_file):
            run_pass_2_logic(raw_text_file, file_id, metadata)
        else:
            print(f"  [Fail] Cannot run Pass 2. Raw text file not found at: {raw_text_file}")
            print("         Please run Pass 1 first (e.g., with --run-pass 1 or --run-pass all).")

def main():
    parser = argparse.ArgumentParser(description="Debug the digitization pipeline on a few random files.")
    parser.add_argument("subject", type=str, help="The subject to search for (e.g., 'Mathematics'). Not used when running only Pass 2.")
    parser.add_argument("num_files", type=int, nargs='?', default=2, help="Number of random files to process. Default: 2.")
    parser.add_argument("--run-pass", type=str, default="all", choices=["1", "2", "all"], help="Specify which pass to run. '1' for vision, '2' for logic, 'all' for both. Default: 'all'.")
    args = parser.parse_args()

    if args.run_pass == '2':
        records = find_staged_files(args.num_files)
    else:
        records = find_random_pdfs(args.subject, args.num_files)

    if not records:
        print("No files found to process. Exiting.")
        return

    for row in records:
        process_file(row, args.run_pass)

if __name__ == "__main__":
    main()