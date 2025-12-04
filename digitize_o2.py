import os
import json
import uuid
import sqlite3
import datetime
from pathlib import Path
from pdf2image import convert_from_path
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- CONFIGURATION ---
INPUT_FOLDER = "./scans"
OUTPUT_FOLDER = "./json_vault"
DB_PATH = "./downloads.db"  # Path to your sqlite DB
MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"

# --- DATABASE HELPER ---
def fetch_metadata_from_db(file_path):
    """
    Queries the downloads.db for metadata based on the filename.
    Opens in Read-Only mode to avoid locking.
    """
    filename = os.path.basename(file_path)
    
    # Defaults
    meta = {
        "institution": "Unknown",
        "class": "Unknown",
        "year": "Unknown",
        "subject": "Unknown",
        "type": "Unknown"
    }

    if not os.path.exists(DB_PATH):
        return meta

    try:
        # URI connection for Read-Only mode
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        # We search by partial path match or exact filename match
        # Adjust query logic if your 'path' column format is strictly absolute
        query = """
            SELECT institution, class, year, subject, type 
            FROM downloads 
            WHERE path LIKE ? OR complete_url LIKE ? 
            LIMIT 1
        """
        search_term = f"%{filename}%"
        cursor.execute(query, (search_term, search_term))
        row = cursor.fetchone()
        
        if row:
            meta["institution"] = row[0] or "Unknown"
            meta["class"] = row[1] or "Unknown"
            meta["year"] = row[2] or "Unknown"
            meta["subject"] = row[3] or "Unknown"
            meta["type"] = row[4] or "Unknown"
            
        conn.close()
    except Exception as e:
        print(f"DB Read Error: {e}")

    return meta

# --- MODEL SETUP ---
print(f"Loading {MODEL_PATH}...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    load_in_4bit=True
)

# Set pixel limits to avoid OOM
min_pixels = 256 * 28 * 28
max_pixels = 1024 * 28 * 28 
processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels)

# --- PROMPT ---
# Simplified prompt: We don't ask for class/year/subject heavily here 
# since we get it from DB, but we ask for 'exam_specifics' (like Set A, Term 1)
SYSTEM_PROMPT = """
You are an expert digitization assistant.
1. Extract questions and answers from the image.
2. If available, extract the specific 'Exam Name' or 'Set Code' (e.g., Half Yearly, Set B).
3. Output purely in JSON format.

Structure:
{
  "exam_specifics": {"exam_name": "...", "set_code": "..."},
  "questions": [
    { "q_no": "1", "text": "...", "marks": "...", "answer": "..." }
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

    # Added repetition_penalty to stop the loop seen in your Page 11 example
    generated_ids = model.generate(**inputs, max_new_tokens=2048, repetition_penalty=1.1)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def clean_json_output(text):
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.endswith("```"): text = text[:-3]
    return text

def main():
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                print(f"Processing: {file}")

                # 1. GET METADATA FROM DB
                db_meta = fetch_metadata_from_db(file_path)
                print(f"  Found DB Metadata: {db_meta}")

                doc_id = str(uuid.uuid4())
                
                # Global Document Container
                final_document = {
                    "document_id": doc_id,
                    "source_file": file,
                    "processed_at": datetime.datetime.now().isoformat(),
                    "metadata": db_meta, # Base metadata from DB
                    "pages": []
                }

                try:
                    images = convert_from_path(file_path)
                    
                    for i, image in enumerate(images):
                        print(f"  - Page {i+1}...")
                        raw_response = process_page(image)
                        
                        try:
                            page_json = json.loads(clean_json_output(raw_response))
                            
                            # Merge extracted exam specifics into global metadata if missing
                            if "exam_specifics" in page_json:
                                for k, v in page_json["exam_specifics"].items():
                                    if v and v != "Unknown" and k not in final_document["metadata"]:
                                        final_document["metadata"][k] = v
                            
                            # Append Questions
                            final_document["pages"].append({
                                "page_number": i + 1,
                                "questions": page_json.get("questions", [])
                            })

                        except json.JSONDecodeError:
                            print(f"    JSON Error Page {i+1}")
                            final_document["pages"].append({
                                "page_number": i + 1,
                                "error": "AI_JSON_PARSE_FAIL", 
                                "raw_text": raw_response
                            })

                    # 2. SAVE OPTIMIZED JSON
                    # Use DB metadata for folder structure
                    year = final_document["metadata"].get("year", "UnknownYear").replace(" ", "_")
                    cls = final_document["metadata"].get("class", "UnknownClass").replace(" ", "_")
                    sub = final_document["metadata"].get("subject", "UnknownSubject").replace(" ", "_")

                    save_dir = Path(OUTPUT_FOLDER) / year / cls / sub
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    final_path = save_dir / f"{doc_id}.json"
                    
                    with open(final_path, 'w', encoding='utf-8') as f:
                        json.dump(final_document, f, indent=2, ensure_ascii=False)
                    
                    print(f"  Saved to {final_path}")

                except Exception as e:
                    print(f"  CRITICAL ERROR: {e}")

if __name__ == "__main__":
    main()
