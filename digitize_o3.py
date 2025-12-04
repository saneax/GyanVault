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
INPUT_ROOT = "output"        # root folder created by your crawler
OUTPUT_VAULT = "json_vault"  # where final JSONs go
DB_PATH = "downloads.db"     # your sqlite database
MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"

# --- DATABASE HELPER ---
def fetch_metadata(file_path):
    """
    Matches the current file path against the 'path' column in downloads.db
    Returns a dictionary of high-confidence metadata.
    """
    # Normalized path string to match DB (e.g., output/2025/X/Subject/file.pdf)
    # We ensure we match the format stored by your crawler
    search_path = str(Path(file_path)) 
    
    meta = {
        "institution": "Unknown", "class": "Unknown", 
        "year": "Unknown", "subject": "Unknown", "type": "Unknown"
    }

    if not os.path.exists(DB_PATH):
        return meta

    try:
        # Open in Read-Only mode (URI syntax) to prevent locking
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        # Exact path match is best, fallback to filename if path differs slightly
        query = """
            SELECT institution, class, year, subject, type 
            FROM downloads 
            WHERE path = ? OR path LIKE ?
            LIMIT 1
        """
        cursor.execute(query, (search_path, f"%{os.path.basename(search_path)}"))
        row = cursor.fetchone()
        
        if row:
            meta = {
                "institution": row[0],
                "class": row[1],
                "year": row[2],
                "subject": row[3],
                "type": row[4]
            }
        conn.close()
    except Exception as e:
        print(f"[DB Error] Could not fetch metadata: {e}")

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

# --- TEACHER PROMPT (Wholesome Answers & MCQs) ---
SYSTEM_PROMPT = """
You are an expert academic teacher and digitizer. Your goal is to digitize this question paper for students.

TASKS:
1. **Extraction**: Read the text exactly. 
2. **MCQ Detection**: If a question has options (A, B, C, D), extract them into a list.
3. **Wholesome Solving**: 
   - Provide a DETAILED, educational answer. 
   - For subjective questions: Write 50-100 words explaining the concept.
   - For Multiple Choice: State the correct option AND explain WHY it is correct.
   - For Math: Show step-by-step calculation.

OUTPUT FORMAT:
Return strictly valid JSON.
{
  "exam_details": "Extract visible exam name/code (e.g. Set-1, Half Yearly)",
  "questions": [
    {
      "q_no": "1",
      "text": "The full question text here...",
      "type": "MCQ",  // or "Subjective"
      "options": ["(a) Option 1", "(b) Option 2", ...], // Empty list if not MCQ
      "marks": "5",
      "answer": "The correct answer is (b).",
      "explanation": "Here is the detailed explanation... [Educational Context]"
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

    # Lower temperature for factual accuracy, penalty to stop loops
    generated_ids = model.generate(**inputs, max_new_tokens=2500, repetition_penalty=1.15, temperature=0.2)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def clean_json_output(text):
    text = text.strip()
    # Aggressive cleanup for markdown blocks
    if "```json" in text:
        text = text.split("```json")[1]
    if "```" in text:
        text = text.split("```")[0]
    return text.strip()

def main():
    # Recursive walk through the 'output' folder structure
    files_to_process = list(Path(INPUT_ROOT).rglob("*.pdf"))
    print(f"Found {len(files_to_process)} PDFs to process.")

    for file_path in files_to_process:
        print(f"\nProcessing: {file_path}")
        
        # 1. Fetch High-Quality Metadata from DB
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
                    page_data = json.loads(clean_json_output(raw_response))
                    
                    # Add page context to questions
                    questions = page_data.get("questions", [])
                    for q in questions:
                        q["page_number"] = i + 1
                        final_document["content"].append(q)

                    # Capture Exam Set Code if found (only need it once)
                    if "exam_details" in page_data and page_data["exam_details"]:
                         if "exam_details" not in final_document["metadata"]:
                             final_document["metadata"]["exam_details"] = page_data["exam_details"]

                except json.JSONDecodeError:
                    print(f"    [JSON Error] Page {i+1} - Text saved as raw_error")
                    final_document["content"].append({
                        "page_number": i + 1,
                        "error": "AI_parsing_failed",
                        "raw_output": raw_response
                    })

            # 2. Save using Crawlers Folder Structure
            # We mirror the input folder structure: json_vault/2025/X/Subject/uuid.json
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

