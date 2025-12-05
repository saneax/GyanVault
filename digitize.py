import sqlite3
import os
import json
import gc
import torch
import hashlib
import re
from pathlib import Path
from pdf2image import convert_from_path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info # Assuming this file exists and is correctly imported
from PIL import Image
import argparse # Added for command-line arguments
import sys # Added for graceful exit

# --- CONFIGURATION ---
DB_PATH = "downloads.db"
ROOT_OUTPUT_DIR = "output"
STAGING_DIR = "./staging2"
FINAL_DIR = "./digitized"
STATE_FILE = os.path.join(STAGING_DIR, "processing_state.json") # New: Path for state file

# --- SAFETY SETTINGS FOR 8GB VRAM ---
OCR_DPI = 300              # Lowered from 300 to 150 (4x less memory)
MAX_IMAGE_DIMENSION = 1024 # Corrected: Cap max width/height to a reasonable single dimension
MAX_PIXELS = 1280 * 28 * 28
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
    return hashlib.md5(file_path.encode()).hexdigest()[:10]

# --- State Management ---
def load_processing_state():
    """Loads the processing state from a JSON file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
            return set(state.get('pass1_completed_ids', [])), set(state.get('pass2_completed_ids', []))
    return set(), set()

def save_processing_state(pass1_completed_ids, pass2_completed_ids):
    """Saves the current processing state to a JSON file."""
    state = {
        'pass1_completed_ids': list(pass1_completed_ids),
        'pass2_completed_ids': list(pass2_completed_ids)
    }
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    print(f"Processing state saved to {STATE_FILE}")

# =========================================================================
# PHASE 1: DISCOVERY & VISION (The "Eye")
# =========================================================================
def run_pass_1_vision(args):
    print("\n" + "="*50)
    print(">>> PASS 1: VISION & TRANSCRIPTION (VRAM SAFE MODE)")
    print("="*50)

    conn = get_db_connection()
    cursor = conn.cursor()
    
    pass1_completed_ids, pass2_completed_ids = load_processing_state()
    files_processed_count = 0

    try:
        # Dynamically build the query based on arguments
        where_clauses = []
        params = []

        if args.subject:
            where_clauses.append("LOWER(subject) = ?")
            params.append(args.subject.lower())
        
        if args.search_term:
            where_clauses.append("(LOWER(path) LIKE ? OR LOWER(pdfs_json) LIKE ?)")
            params.extend([f'%{args.search_term.lower()}%', f'%{args.search_term.lower()}%'])

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Get total count to validate offset
        count_query = f"SELECT COUNT(*) FROM downloads WHERE {where_sql}"
        cursor.execute(count_query, params)
        total_records = cursor.fetchone()[0]

        if args.offset >= total_records and total_records > 0:
            print(f"!! Warning: Offset ({args.offset}) is greater than or equal to the total matching records ({total_records}). No files to process in Pass 1.")
            conn.close()
            return

        limit_sql = "LIMIT ?"
        params.append(args.max_files if args.max_files > 0 else -1)
        offset_sql = "OFFSET ?"
        params.append(args.offset)

        query = f"SELECT complete_url, year, class, subject, path, pdfs_json FROM downloads WHERE {where_sql} {limit_sql} {offset_sql}"

        cursor.execute(query, params)
        records = cursor.fetchall()
        
        print(f"DEBUG: Found {len(records)} records in database matching criteria.")
        
    except sqlite3.OperationalError as e:
        print(f"!! DB Error: {e}")
        return

    # CHANGE 1: Use the 7B model instead of 2B
    model_path = "Qwen/Qwen2-VL-7B-Instruct"
    
    # CHANGE 2: Load in 4-bit to fit in 8GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = None # Initialize model to None
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            quantization_config=bnb_config,
            device_map="auto", 
            attn_implementation="flash_attention_2" 
        )
    except Exception as e: # Catch specific exception for clarity
        print(f">> Flash Attention 2 not available ({e}), falling back to SDPA.")
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="sdpa"
            )
        except Exception as e_sdpa:
            print(f"!! Error loading model with SDPA either: {e_sdpa}")
            cleanup_gpu()
            return # Exit if model can't be loaded

    # Use a min_pixels / max_pixels strategy in the processor
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=MAX_PIXELS)
    # REPLACE THE LOOP IN 'run_pass_1_vision' WITH THIS:
    print(f"DEBUG: Found {len(records)} records in database.")

    try: # Main processing loop wrapped in try-except for KeyboardInterrupt
        for row in records:
            # --- File Discovery Logic ---
            target_files = []
            pdfs_json_str = row['pdfs_json']
            main_path = row['path']
            
            if pdfs_json_str and pdfs_json_str != "[]":
                try:
                    pdf_list = json.loads(pdfs_json_str)
                    target_files = [p['file'] for p in pdf_list]
                except json.JSONDecodeError:
                    target_files = []
            
            if not target_files and main_path and main_path.lower().endswith('.pdf'):
                if str(main_path).startswith("output/"):
                     target_files = [str(main_path).replace("output/", "", 1)]
                else:
                     target_files = [main_path]
            if not target_files:
                continue
    
            # --- Processing Loop ---
            for rel_path in target_files:
                file_id = get_unique_file_id(rel_path)
    
                full_pdf_path = os.path.join(ROOT_OUTPUT_DIR, rel_path)
                raw_text_path = os.path.join(STAGING_DIR, f"{file_id}_raw.txt")
    
                # Path fallback check
                if not os.path.exists(full_pdf_path):
                        # Check if it's already in ROOT_OUTPUT_DIR/output/
                        full_pdf_path_alt = os.path.join(ROOT_OUTPUT_DIR, "output", rel_path)
                        if os.path.exists(full_pdf_path_alt):
                            full_pdf_path = full_pdf_path_alt
                        else:
                            print(f"   !! File not found: {rel_path}")
                            continue
                
                # Stop if we have considered the max number of files, regardless of status
                if args.max_files > 0 and files_processed_count >= args.max_files:
                    print(f"-> Reached file consideration limit ({args.max_files}). Stopping Pass 1.")
                    return # Exit the function
                
                # Check if raw text exists and is up-to-date
                if os.path.exists(raw_text_path):
                    pdf_mtime = os.path.getmtime(full_pdf_path)
                    raw_text_mtime = os.path.getmtime(raw_text_path)
                    if pdf_mtime < raw_text_mtime:
                        print(f"-> Skipping (raw text is up-to-date): {rel_path}")
                        files_processed_count += 1 # Count this file towards the limit
                        continue
                    else:
                        print(f"-> Re-processing (source PDF has been updated): {rel_path}")
                print(f"-> Processing: {rel_path}")
    
                # 1. Convert PDF to Images (Low DPI)
                try:
                    images = convert_from_path(full_pdf_path, dpi=OCR_DPI)
                except Exception as e:
                    print(f"   !! PDF Conversion Error for {rel_path}: {e}")
                    continue
                full_text_content = ""
    
                # 2. OCR Each Page
                for i, image in enumerate(images):
                    # SAFEGUARD: Resize if too big
                    if max(image.size) > MAX_IMAGE_DIMENSION:
                        image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
                    
                    # Save temp image for the model to read
                    temp_img_path = os.path.join(STAGING_DIR, "temp_vision.png")
                    image.save(temp_img_path)
                    # IN: run_pass_1_vision()
    
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": temp_img_path},
                            {"type": "text", "text": (
                                "Transcribe this page line-by-line.\n"
                                "1. If you see a diagram, graph, or geometric figure, describe it in detail "
                                "inside square brackets like this: [DIAGRAM: A triangle with sides 5cm...].\n"
                                "2. If you see math formulas, write them in LaTeX format.\n"
                                "3. Do not ignore the Hindi text, transcribe it exactly as seen."
                            )}
                        ]
                    }]
                    
                    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    
                    inputs = processor(
                        text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
                    ).to("cuda")
    
                    # Generate
                    with torch.no_grad(): # Disable gradient calc to save RA
    
                        generated_ids = model.generate(**inputs, max_new_tokens=1024)
                    
    
                    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
                    page_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
                    
                    full_text_content += f"\n--- PAGE {i+1} ---\n{page_text}\n"
                    
                    # CRITICAL: Clear VRAM after every page
                    del inputs, generated_ids, image_inputs
                    cleanup_gpu()
                # 3. Save Results
                with open(raw_text_path, "w", encoding="utf-8") as f:
                    f.write(full_text_content)
                
                subject = row['subject']
                # If subject is generic, try to guess from the filename
                if subject and subject.lower() == 'download':
                    stem = Path(rel_path).stem
                    # Remove common codes, numbers, and series letters from the start of the filename
                    guessed_subject = re.sub(r'^[0-9\s\(\)A-Z_-]+', '', stem, flags=re.IGNORECASE).strip()
                    if guessed_subject: # Only replace if we found something
                        subject = guessed_subject

                meta = {
                    "subject": subject, # Use the potentially improved subject
                    "class": row['class'],
                    "year": row['year'],
                    "original_path": rel_path
                }
                with open(os.path.join(STAGING_DIR, f"{file_id}_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f)
                
                print(f"   [Saved] {file_id}_raw.txt")
                pass1_completed_ids.add(file_id) # Mark as completed in state
                files_processed_count += 1 # Increment after successful processing

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Saving state and exiting Pass 1 gracefully.")
    except Exception as e:
        print(f"\nAn unexpected error occurred in Pass 1: {e}")
    finally:
        if 'model' in locals() and model: # Ensure model is deleted only if it was loaded
            del model
        if 'processor' in locals() and processor: # Ensure processor is deleted only if it was loaded
            del processor
        cleanup_gpu()
        save_processing_state(pass1_completed_ids, pass2_completed_ids)



# =========================================================================
# PHASE 2: LOGIC & JSON FORMATTING (The "Brain")
# =========================================================================
def run_pass_2_logic(args):
    print("\n" + "="*50)
    print(">>> PASS 2: LOGIC & JSON FORMATTING")
    print("="*50)

    pass1_completed_ids, pass2_completed_ids = load_processing_state()
    files_processed_count = 0

    pending_files_for_pass2 = []
    for f in os.listdir(STAGING_DIR):
        if f.endswith("_raw.txt"):
            file_id = f.split("_")[0]
            meta_path = os.path.join(STAGING_DIR, f"{file_id}_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as mf:
                    meta = json.load(mf)

                if file_id not in pass1_completed_ids:
                    continue

                if file_id in pass2_completed_ids:
                    print(f"-> Skipping (Pass 2 completed): {file_id}")
                    continue
                
                final_path_check = os.path.join(FINAL_DIR, f"{file_id}_{meta.get('subject', 'unknown')}.json")
                if os.path.exists(final_path_check):
                    print(f"-> Skipping (final JSON already exists): {file_id}")
                    continue
                pending_files_for_pass2.append((f, meta, file_id))

    if not pending_files_for_pass2:
        print(">>> No pending files for Pass 2.")
        return

    print(">>> Loading Logic Model (Qwen2-7B-Instruct)...")
    model_id = "Qwen/Qwen2-7B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto"
        )
    except Exception as e:
        print(f"!! Error loading Logic Model: {e}")
        cleanup_gpu()
        return

    try:
        for filename, meta, file_id in pending_files_for_pass2:
            if args.max_files > 0 and files_processed_count >= args.max_files:
                print(f"-> Reached maximum number of files to process ({args.max_files}). Stopping Pass 2.")
                return

            print(f"-> Structuring JSON for {meta.get('subject')} (File ID: {file_id})")

            with open(os.path.join(STAGING_DIR, filename), "r", encoding="utf-8") as f:
                raw_text = f.read()

            # Chunk the raw text if it's too long to avoid token overflow
            # Split by "PAGE" markers or by question numbers
            text_chunks = []
            current_chunk = ""
            chunk_size = 3000  # characters per chunk
            
            for line in raw_text.split('\n'):
                if len(current_chunk) > chunk_size and ('---' in line or re.match(r'^\d+\.', line)):
                    text_chunks.append(current_chunk)
                    current_chunk = line + "\n"
                else:
                    current_chunk += line + "\n"
            if current_chunk:
                text_chunks.append(current_chunk)

            # Process in chunks and merge results
            all_questions = []
            chunk_metadata = None
            
            for chunk_idx, chunk_text in enumerate(text_chunks):
                print(f"   Processing chunk {chunk_idx + 1}/{len(text_chunks)}...")
                
                prompt = rf"""
You are a CBSE Question Paper digitization expert.

CONTEXT:
Subject: {meta['subject']}
Class: {meta['class']}
Year: {meta['year']}
Source File: {meta.get('original_path', 'N/A')}
Chunk: {chunk_idx + 1}/{len(text_chunks)}

TASK:
Convert the OCR text into clean JSON. Output ONLY valid JSON, no markdown, no explanations.

RULES:
1. **LANGUAGE FILTER:** Extract ONLY English questions. Ignore Hindi entirely.
2. **DIAGRAMS:** Include `[DIAGRAM: ...]` descriptions or set to "N/A".
3. **JSON SYNTAX:** All backslashes in LaTeX must be escaped: `\\sin` → `\\\\sin`
4. **INCOMPLETE QUESTIONS:** If a question is cut off mid-sentence, DO NOT include it. Only include complete questions.
5. **GENERATE ANSWERS:** Provide step-by-step explanations.
6. **MCQ OPTIONS:** Capture (A), (B), (C), (D) in an options object.

RAW OCR TEXT:
{chunk_text[:2500]}

OUTPUT ONLY THIS JSON (no other text):
{{
    "metadata": {{"subject": "{meta['subject']}", "year": "{meta['year']}", "refined_subject": ""}},
    "questions": []
}}

If there are no complete questions in this chunk, return an empty questions array.
"""

                messages = [
                    {"role": "system", "content": "Output ONLY valid JSON. No markdown, no explanations."},
                    {"role": "user", "content": prompt}
                ]

                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([text], padding=True, return_tensors="pt").to("cuda")
                
                # Add explicit attention mask
                if 'attention_mask' not in inputs:
                    inputs['attention_mask'] = (inputs.input_ids != tokenizer.pad_token_id).int()
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids, 
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=1500,  # Reduced to force shorter, complete responses
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
                json_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                final_name = f"{file_id}_{meta.get('subject', 'unknown')}.json"
                final_path = os.path.join(FINAL_DIR, final_name)
                
                try:
                    # Extract JSON with more robust parsing
                    start_index = json_str.find('{')
                    end_index = json_str.rfind('}')
                    
                    if start_index == -1 or end_index == -1 or end_index <= start_index:
                        print(f"   [Warn] No valid JSON in chunk {chunk_idx + 1}. Skipping.")
                        continue
                    
                    clean_json = json_str[start_index : end_index + 1]
                    clean_json = re.sub(r",\s*([}\]])", r"\1", clean_json)
                    
                    # Attempt to parse and validate
                    parsed_json = json.loads(clean_json)
                    
                    # Validate questions are complete
                    if "questions" in parsed_json:
                        validated_questions = []
                        for q in parsed_json["questions"]:
                            # Check if question has required fields
                            if all(k in q for k in ["q_no", "text", "marks"]):
                                # Check if text ends properly (not cut off)
                                if q["text"].strip().endswith((":", "?", ".")):
                                    validated_questions.append(q)
                                else:
                                    print(f"   [Warn] Skipping incomplete question {q.get('q_no', '?')}")
                        
                        all_questions.extend(validated_questions)
                    
                    if "metadata" in parsed_json:
                        chunk_metadata = parsed_json["metadata"]
                    
                except json.JSONDecodeError as e:
                    print(f"   [Warn] JSON parse error in chunk {chunk_idx + 1}: {e}. Saving to .err file.")
                    with open(final_path + f".err.chunk{chunk_idx}", "w", encoding="utf-8") as f:
                        f.write(json_str)
                except Exception as e:
                    print(f"   [Warn] Error processing chunk {chunk_idx + 1}: {e}")
                
                del inputs, outputs
                cleanup_gpu()
            
            # Merge all chunks into final output
            if all_questions:
                final_output = {
                    "metadata": chunk_metadata or {
                        "subject": meta['subject'],
                        "year": meta['year'],
                        "refined_subject": meta.get('subject', '')
                    },
                    "questions": all_questions
                }
                
                with open(final_path, "w", encoding="utf-8") as f:
                    json.dump(final_output, f, indent=2, ensure_ascii=False)
                
                print(f"   [Success] Saved {final_name} with {len(all_questions)} valid questions")
                pass2_completed_ids.add(file_id)
                files_processed_count += 1
            else:
                print(f"   [Fail] No valid questions extracted for {file_id}. Saving raw output.")
                with open(final_path + ".err", "w", encoding="utf-8") as f:
                    f.write(json_str)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Saving state and exiting Pass 2 gracefully.")
    except Exception as e:
        print(f"\nAn unexpected error occurred in Pass 2: {e}")
    finally:
        if 'model' in locals() and model:
            del model
        if 'tokenizer' in locals() and tokenizer:
            del tokenizer
        cleanup_gpu()
        save_processing_state(pass1_completed_ids, pass2_completed_ids)

def query_and_print(args):
    """Queries the database based on provided arguments and prints the results without processing."""
    print("\n" + "="*50)
    print(">>> QUERY-ONLY MODE")
    print("="*50)
    conn = get_db_connection()
    cursor = conn.cursor()

    where_clauses = []
    params = []

    # Build WHERE clause from arguments
    if args.subject:
        where_clauses.append("LOWER(subject) = ?")
        params.append(args.subject.lower())
    
    if args.search_term:
        where_clauses.append("(LOWER(path) LIKE ? OR LOWER(pdfs_json) LIKE ?)")
        params.extend([f'%{args.search_term.lower()}%', f'%{args.search_term.lower()}%'])

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    # First, get the total count for the given criteria to validate offset
    count_query = f"SELECT COUNT(*) FROM downloads WHERE {where_sql}"
    cursor.execute(count_query, params)
    total_records = cursor.fetchone()[0]

    if args.offset >= total_records:
        print(f"!! Warning: Offset ({args.offset}) is greater than or equal to the total matching records ({total_records}). No files to process.")
        conn.close()
        return

def main():
    parser = argparse.ArgumentParser(description="Digitize PDF documents in two passes.")
    parser.add_argument('--max_files', type=int, default=-1,
                        help="Maximum number of unique PDF files to process in each pass. "
                             "Set to -1 for no limit. Default: -1.")
    parser.add_argument('--offset', type=int, default=0,
                        help="Number of records to skip in the database query. Default: 0.")
    parser.add_argument('--subject', type=str, default=None,
                        help="Filter records by a specific subject (case-insensitive).")
    parser.add_argument('--search_term', type=str, default='mathematics',
                        help="An arbitrary term to search for in the 'path' or 'pdfs_json' fields. "
                             "Default: 'mathematics'.")
    parser.add_argument('--query_only', action='store_true',
                        help="If set, the script will only query the database with the given filters, "
                             "print the matching files, and then exit without processing.")
    args = parser.parse_args()

    if args.query_only:
        query_and_print(args)
        return

    print(f"Starting digitization process with the following settings:")
    print(f"  - Subject: {args.subject or 'Any'}")
    print(f"  - Search Term: '{args.search_term}'")
    print(f"  - Max Files: {args.max_files if args.max_files > 0 else 'No Limit'}")
    print(f"  - Offset: {args.offset}")

    # run_pass_1_vision(args)
    run_pass_2_logic(args)

if __name__ == "__main__":
    main()
