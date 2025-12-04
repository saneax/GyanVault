import os
import json
import uuid
import sqlite3
import datetime
import re
from pathlib import Path
from pdf2image import convert_from_path
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- CONFIGURATION ---
INPUT_ROOT = "output"        
OUTPUT_VAULT = "json_vault"  
DB_PATH = "downloads.db"     
MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"

# --- METADATA HELPERS ---
def clean_subject_from_filename(filename):
    """
    Extracts subject from filenames like '31-3-1_Science.pdf' -> 'Science'
    Removes standard prefixes like '101_', 'Set-1', etc.
    """
    name = Path(filename).stem
    # Remove common prefixes like '31-3-1_' or '102_' using regex
    # Matches starting digits/dashes/underscores
    cleaned = re.sub(r'^[\d\-\.]+_', '', name) 
    return cleaned.replace("_", " ").strip()

def parse_path_metadata(file_path):
    """
    Fallback: Reads metadata from the folder structure itself.
    Expected: output/YEAR/CLASS/SUBJECT/filename.pdf
    """
    p = Path(file_path)
    parts = p.parts
    
    meta = {
        "institution": "CBSE", # Safe default for this folder structure
        "class": "Unknown",
        "year": "Unknown",
        "subject": "Unknown",
        "type": "Question Paper"
    }

    # We expect depth >= 4 (output, year, class, subject, file)
    try:
        # Walk backwards from the file
        if len(parts) >= 2:
            # Check if parent folder is generic 'Download'
            folder_name = parts[-2]
            if folder_name.lower() in ['download', 'downloads']:
                # Use filename for subject
                meta['subject'] = clean_subject_from_filename(parts[-1])
            else:
                meta['subject'] = folder_name
        
        if len(parts) >= 3:
            meta['class'] = parts[-3]
            
        if len(parts) >= 4:
            meta['year'] = parts[-4]
            
    except Exception as e:
        print(f"  [Path Parse Error] {e}")
        
    return meta

def fetch_metadata(file_path):
    """
    1. Try DB match.
    2. If DB returns Unknown, use Path parsing.
    """
    search_path = str(Path(file_path)) 
    
    # Default to Path Metadata first (Weakest source)
    meta = parse_path_metadata(file_path)
    
    if not os.path.exists(DB_PATH):
        return meta

    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        query = """
            SELECT institution, class, year, subject, type 
            FROM downloads 
            WHERE path = ? OR path LIKE ?
            LIMIT 1
        """
        cursor.execute(query, (search_path, f"%{os.path.basename(search_path)}"))
        row = cursor.fetchone()
        
        if row:
            # Only overwrite if DB has real data (not None/Empty)
            if row[0]: meta["institution"] = row[0]
            if row[1]: meta["class"] = row[1]
            if row[2]: meta["year"] = row[2]
            if row[3]: meta["subject"] = row[3]
            if row[4]: meta["type"] = row[4]
            
        conn.close()
    except Exception as e:
        print(f"  [DB Error] {e}")

    return meta

# --- MODEL SETUP ---
print(f"Loading {MODEL_PATH}...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    load_in_4bit=True
)
min_pixels = 256 * 28 * 28
max_pixels = 1024 * 28 * 28 
processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels)

# --- TEACHER PROMPT ---
SYSTEM_PROMPT = """
You are an expert academic teacher. Digitize this question paper.

TASKS:
1. Extract questions exactly.
2. If MCQ, put options in a list ["(A)...", "(B)..."].
3. Provide a DETAILED, wholesome answer. Explain concepts clearly.

OUTPUT JSON FORMAT:
{
  "exam_details": "Set-1 / Term-2",
  "questions": [
    {
      "q_no": "1",
      "text": "...",
      "type": "MCQ", 
      "options": ["(A) ...", "(B) ..."],
      "marks": "1",
      "answer": "Correct option is (A).",
      "explanation": "Detailed explanation..."
    }
  ]
}
"""

def process_page(image):
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": SYSTEM_PROMPT}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=2500, repetition_penalty=1.15, temperature=0.2)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]

def clean_json_output(text):
    text = text.strip()
    if "```json" in text: text = text.split("```json")[1]
    if "```" in text: text = text.split("```")[0]
    return text.strip()

def main():
    files_to_process = list(Path(INPUT_ROOT).rglob("*.pdf"))
    print(f"Found {len(files_to_process)} PDFs to process.")

    for file_path in files_to_process:
        print(f"\nProcessing: {file_path}")
        
        # 1. Fetch Metadata (DB + Path Fallback)
        db_meta = fetch_metadata(file_path)
        print(f"  Metadata: {db_meta}")

        doc_id = str(uuid.uuid4())
        final_document = {
            "document_id": doc_id,
            "source_file": file_path.name,
            "processed_at": datetime.datetime.now().isoformat(),
            "metadata": db_meta, 
            "content": []
        }

        try:
            images = convert_from_path(file_path)
            
            for i, image in enumerate(images):
                print(f"  - Analyzing Page {i+1}...")
                raw_response = process_page(image)
                
                try:
                    clean_txt = clean_json_output(raw_response)
                    page_data = json.loads(clean_txt)
                    
                    # --- CRITICAL FIX: Handle List vs Dict ---
                    if isinstance(page_data, list):
                        # The AI forgot the wrapper, so we add it manually
                        page_data = {"questions": page_data}
                    
                    # Add page context
                    questions = page_data.get("questions", [])
                    if questions:
                        for q in questions:
                            q["page_number"] = i + 1
                            final_document["content"].append(q)

                    # Capture Exam Details
                    if "exam_details" in page_data and page_data["exam_details"]:
                         if "exam_details" not in final_document["metadata"]:
                             final_document["metadata"]["exam_details"] = page_data["exam_details"]

                except json.JSONDecodeError:
                    print(f"    [JSON Error] Page {i+1} - Raw text saved.")
                    final_document["content"].append({
                        "page_number": i + 1,
                        "error": "AI_parsing_failed",
                        "raw_output": raw_response
                    })

            # 2. Save
            relative_folder = file_path.parent.relative_to(INPUT_ROOT)
            save_dir = Path(OUTPUT_VAULT) / relative_folder
            save_dir.mkdir(parents=True, exist_ok=True)
            
            final_path = save_dir / f"{file_path.stem}_{doc_id[:8]}.json"
            
            with open(final_path, 'w', encoding='utf-8') as f:
                json.dump(final_document, f, indent=2, ensure_ascii=False)
            
            print(f"  Saved: {final_path}")

        except Exception as e:
            print(f"  CRITICAL FAIL: {e}")

if __name__ == "__main__":
    main()

