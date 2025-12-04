import os
import json
import uuid
import sqlite3
import datetime
import re
import torch
from pathlib import Path
from pdf2image import convert_from_path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- CONFIGURATION ---
INPUT_ROOT = "output"        
OUTPUT_VAULT = "json_vault"  
DB_PATH = "downloads.db"     
MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"

# --- METADATA ENGINE ---
def clean_subject_from_filename(filename):
    """
    Extracts subject from filenames like '16_Manipuri.pdf' -> 'Manipuri'
    """
    name = Path(filename).stem
    # Remove leading digits and underscores (e.g., "16_", "31-1-1_")
    cleaned = re.sub(r'^[\d\-\.]+_', '', name) 
    return cleaned.replace("_", " ").strip()

def get_smart_metadata(file_path):
    """
    Intelligent metadata extraction.
    Prioritizes Filename for Subject if folder is generic.
    """
    p = Path(file_path)
    parts = p.parts
    
    # 1. Default Extraction from Path
    meta = {
        "institution": "CBSE", 
        "class": "Unknown",
        "year": "Unknown",
        "subject": "Unknown",
        "type": "Question Paper",
        "language_hint": "English" # Default
    }

    try:
        # Extract Year/Class if available in path
        if len(parts) >= 4: meta['year'] = parts[-4]
        if len(parts) >= 3: meta['class'] = parts[-3]
        
        # Smart Subject Logic
        folder_subject = parts[-2]
        if folder_subject.lower() in ['download', 'downloads', 'new', 'question_paper']:
            # Folder is generic, force use filename
            meta['subject'] = clean_subject_from_filename(parts[-1])
        else:
            meta['subject'] = folder_subject
            
        # Language Hinting (Simple heuristic)
        if meta['subject'].lower() in ['manipuri', 'hindi', 'sanskrit', 'urdu', 'bengali', 'assamese']:
             meta['language_hint'] = meta['subject']
             
    except Exception as e:
        print(f"  [Meta Error] {e}")
        
    # 2. Database Overlay (Optional: If you have the DB active)
    # [Code omitted for brevity, logic remains same as v3]
    
    return meta

# --- MODEL SETUP (SPEED OPTIMIZED) ---
print(f"Loading {MODEL_PATH}...")

# OPTIMIZATION: Use bfloat16 instead of 4bit if you have 8GB VRAM.
# The 2B model is small (~4GB in fp16). Running in native precision is FASTER than 4-bit decoding.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16, # FASTER than auto/4bit on RTX 40 series
    device_map="cuda",
    # attn_implementation="flash_attention_2" # Uncomment if you have flash-attn installed for 2x speed
)

# Resolution Optimization:
# Min 512 to ensure text readability, Max 1024 to keep speed high. 
# Anything larger than 1280 slows down generation significantly.
min_pixels = 512 * 28 * 28
max_pixels = 1280 * 28 * 28 
processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels)

# --- STRICT TEACHER PROMPT ---
SYSTEM_PROMPT = """
You are an expert academic digitizer. 

STRICT RULES:
1. **Ignore Instructions**: Do NOT capture text like "Candidates must write code...", "Please check this paper...", "Time allowed...", "General Instructions".
2. **Capture Only Questions**: Only extract actual exam questions and their sub-parts.
3. **Language**: The paper may be in {language}. If you cannot translate, transcribe the text exactly in the original script.
4. **Wholesome Answers**: Provide educational explanations for answers.

OUTPUT JSON FORMAT:
{{
  "exam_details": "Set-4 / Q.P Code...",
  "questions": [
    {{
      "q_no": "1",
      "text": "Actual question text...",
      "type": "MCQ", 
      "options": ["(A) ...", "(B) ..."],
      "marks": "1",
      "answer": "Correct option is...",
      "explanation": "..."
    }}
  ]
}}
"""

def process_page(image, language_hint):
    # Dynamic prompt injection
    prompt = SYSTEM_PROMPT.format(language=language_hint)
    
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
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

    # OPTIMIZATION: Reduced max_new_tokens to 1500 to prevent infinite loops on garbage text
    generated_ids = model.generate(**inputs, max_new_tokens=1500, repetition_penalty=1.2, temperature=0.2)
    
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
    print(f"Found {len(files_to_process)} PDFs.")

    for file_path in files_to_process:
        print(f"\nProcessing: {file_path}")
        
        # 1. Smart Metadata
        meta = get_smart_metadata(file_path)
        print(f"  Subject: {meta['subject']} | Class: {meta['class']}")

        doc_id = str(uuid.uuid4())
        final_document = {
            "document_id": doc_id,
            "source_file": file_path.name,
            "processed_at": datetime.datetime.now().isoformat(),
            "metadata": meta, 
            "content": []
        }

        try:
            images = convert_from_path(file_path)
            
            for i, image in enumerate(images):
                print(f"  - Page {i+1}...")
                raw_response = process_page(image, meta['language_hint'])
                
                try:
                    clean_txt = clean_json_output(raw_response)
                    page_data = json.loads(clean_txt)
                    
                    if isinstance(page_data, list): page_data = {"questions": page_data}
                    
                    questions = page_data.get("questions", [])
                    if questions:
                        for q in questions:
                            q["page_number"] = i + 1
                            final_document["content"].append(q)
                            
                    if "exam_details" in page_data:
                         final_document["metadata"]["exam_details"] = page_data["exam_details"]

                except json.JSONDecodeError:
                    print(f"    [Warning] Page {i+1} JSON failed. Saving raw text to avoid data loss.")
                    # Fallback: Capture raw text so you don't lose the page content
                    final_document["content"].append({
                        "page_number": i + 1,
                        "type": "raw_text_fallback",
                        "raw_content": raw_response
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

