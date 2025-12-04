import os
import json
import uuid
import re
import torch
import datetime
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoModel, AutoTokenizer

# --- CONFIGURATION ---
INPUT_ROOT = "output"        
OUTPUT_VAULT = "json_vault"  
# We use the specific INT4 version which fits perfectly in your 8GB VRAM
# This model downloads automatically (~5-6GB)
MODEL_PATH = "openbmb/MiniCPM-V-2_6-int4"

# --- SMART METADATA ENGINE ---
def get_indic_metadata(file_path):
    """
    Enhanced metadata extraction for Indic contexts.
    Detects if the subject implies a specific script (e.g., 'Hindi' -> Devanagari).
    """
    p = Path(file_path)
    parts = p.parts
    
    meta = {
        "institution": "CBSE", 
        "class": "Unknown",
        "year": "Unknown",
        "subject": "Unknown",
        "script_hint": "Latin/English" # Default
    }

    try:
        # Extract Path Info
        if len(parts) >= 4: meta['year'] = parts[-4]
        if len(parts) >= 3: meta['class'] = parts[-3]
        
        # Subject Extraction
        folder_subject = parts[-2]
        if folder_subject.lower() in ['download', 'downloads', 'new']:
            # Clean filename: "16_Manipuri.pdf" -> "Manipuri"
            name = p.stem
            cleaned = re.sub(r'^[\d\-\.]+_', '', name).replace("_", " ").strip()
            meta['subject'] = cleaned
        else:
            meta['subject'] = folder_subject
            
        # Script/Language Detection map
        subj_lower = meta['subject'].lower()
        if 'hindi' in subj_lower or 'sanskrit' in subj_lower:
            meta['script_hint'] = "Devanagari (Hindi/Sanskrit)"
        elif 'manipuri' in subj_lower:
            meta['script_hint'] = "Bengali Script or Meitei Mayek"
        elif 'bengali' in subj_lower:
             meta['script_hint'] = "Bengali"
        elif 'urdu' in subj_lower:
             meta['script_hint'] = "Urdu"
        elif 'punjabi' in subj_lower:
             meta['script_hint'] = "Gurmukhi"
             
    except Exception as e:
        print(f"  [Meta Error] {e}")
    
    return meta

# --- MODEL SETUP ---
print(f"Loading {MODEL_PATH} (High-Res Indic Mode)...")
# MiniCPM-V requires trust_remote_code=True
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.eval()

# --- PROMPT FOR INDIC TEXT ---
# We explicitly tell the model NOT to translate, but to TRANSCRIBE.
INDIC_PROMPT = """
Analyze this exam paper image. The text may contain {script} script.

STRICT INSTRUCTIONS:
1. **Transcribe Exactly**: Do NOT translate the text. If the text is in {script}, output it in {script} (Unicode).
2. **Ignore Instructions**: Ignore text like "Time allowed", "Marks", "General Instructions".
3. **Capture Questions**: Extract questions, their numbers, and options.
4. **Formatting**: If the text is garbled or unclear, use the placeholder "[Unreadable]".

Output PURE JSON format:
{{
  "exam_code": "Set-A...",
  "questions": [
    {{
      "q_no": "1",
      "text": "Question text in original script...",
      "type": "MCQ", 
      "options": ["(A) Option in orig script", "(B) ..."], 
      "marks": "1",
      "answer": "Correct option",
      "explanation": "Explanation in English (if possible) or original script."
    }}
  ]
}}
"""

def process_page_minicpm(image, script_hint):
    # MiniCPM handles high-res images internally.
    # We pass the image directly.
    
    prompt = INDIC_PROMPT.format(script=script_hint)
    
    msgs = [{'role': 'user', 'content': [image, prompt]}]

    # Speed Optimization: decode_strategy parameters
    try:
        response = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True, 
            temperature=0.2, # Low temp for factual accuracy on OCR
            max_new_tokens=2048
        )
        return response
    except Exception as e:
        return f"ERROR: {str(e)}"

def clean_json(text):
    text = text.strip()
    if "```json" in text: text = text.split("```json")[1]
    if "```" in text: text = text.split("```")[0]
    return text.strip()

def main():
    files = list(Path(INPUT_ROOT).rglob("*.pdf"))
    print(f"Found {len(files)} PDFs.")

    for file_path in files:
        print(f"\nProcessing: {file_path}")
        meta = get_indic_metadata(file_path)
        print(f"  Subject: {meta['subject']} | Script Hint: {meta['script_hint']}")

        doc_id = str(uuid.uuid4())
        doc_data = {
            "id": doc_id,
            "source": file_path.name,
            "meta": meta,
            "content": []
        }

        try:
            # Convert PDF to images
            # DPI=200 is usually enough for MiniCPM, but 300 is safer for complex scripts
            images = convert_from_path(file_path, dpi=300) 
            
            for i, img in enumerate(images):
                print(f"  - Page {i+1}...")
                
                raw_res = process_page_minicpm(img, meta['script_hint'])
                
                # Validation & Fallback
                try:
                    cleaned = clean_json(raw_res)
                    parsed = json.loads(cleaned)
                    
                    # Normalization
                    if isinstance(parsed, list): parsed = {"questions": parsed}
                    
                    qs = parsed.get("questions", [])
                    for q in qs:
                        q['page'] = i + 1
                        doc_data["content"].append(q)
                        
                except json.JSONDecodeError:
                    print(f"    [Warn] JSON fail on Page {i+1}. Saving Raw.")
                    doc_data["content"].append({
                        "page": i + 1,
                        "type": "raw_indic_text",
                        "content": raw_res
                    })
            
            # Save
            rel_dir = file_path.parent.relative_to(INPUT_ROOT)
            save_path = Path(OUTPUT_VAULT) / rel_dir
            save_path.mkdir(parents=True, exist_ok=True)
            
            final_file = save_path / f"{file_path.stem}_{doc_id[:6]}.json"
            with open(final_file, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {final_file}")

        except Exception as e:
            print(f"  CRITICAL FAIL: {e}")

if __name__ == "__main__":
    main()

