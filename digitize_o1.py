import os
import json
import uuid
from pathlib import Path
from pdf2image import convert_from_path
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- CONFIGURATION ---
INPUT_FOLDER = "./scans"
OUTPUT_FOLDER = "./json_vault"

# CHANGE 1: Use the 2B model. It fits easily in 8GB VRAM.
MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct" 

# --- SETUP MODEL ---
print(f"Loading {MODEL_PATH}...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    # We can likely run 2B in full precision (bf16) on 8GB, 
    # but 4-bit is safer if you have other apps open.
    # remove load_in_4bit=True if you want slightly higher accuracy
    load_in_4bit=True 
)

# CHANGE 2: Limit resolution to prevent memory spikes on large scans
# min_pixels: ensures small text is legible
# max_pixels: prevents the image from creating too many tokens (OOM killer)
min_pixels = 256 * 28 * 28
max_pixels = 1024 * 28 * 28 

processor = AutoProcessor.from_pretrained(
    MODEL_PATH, 
    min_pixels=min_pixels, 
    max_pixels=max_pixels
)
# --- THE PROMPT ---
# We force the AI to return ONLY JSON.
SYSTEM_PROMPT = """
You are an expert archivist and professor. 
Analyze the provided image of a question paper.
1. Extract Metadata: Institution name, Class/Standard, Year, Subject, Exam Name. Use "Unknown" if not visible.
2. Extract Questions: Read the text exactly as written.
3. Solve Questions: Provide a concise, correct answer for each question based on your knowledge.
4. Output strictly in valid JSON format with no Markdown formatting (```json).
   
Expected JSON Structure:
{
  "metadata": {
    "institution": "string",
    "class": "string",
    "year": "string",
    "subject": "string",
    "exam_name": "string"
  },
  "content": [
    {
      "q_no": "1",
      "question_text": "...",
      "marks": "...",
      "generated_answer": "..."
    }
  ]
}
"""

def process_image(image, image_path):
    # Construct conversation
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

def clean_json_output(text):
    # Sometimes LLMs add ```json at start and ``` at end. Remove them.
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    return text

def main():
    # Create output directory
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    # Walk through input folder
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                
                try:
                    # Convert PDF to Images (One image per page)
                    # We process page by page to avoid context limit overflow
                    images = convert_from_path(file_path)
                    
                    paper_data = []
                    
                    for i, image in enumerate(images):
                        print(f"  - Analyzing Page {i+1}...")
                        response_text = process_image(image, file_path)
                        
                        try:
                            clean_json = clean_json_output(response_text)
                            page_data = json.loads(clean_json)
                            # Tag page number
                            page_data['page_number'] = i + 1
                            paper_data.append(page_data)
                        except json.JSONDecodeError:
                            print(f"    Error: AI did not return valid JSON for page {i+1}. Saving raw text.")
                            paper_data.append({"page_number": i+1, "raw_error_text": response_text})

                    # Construct Final JSON filename based on first page metadata if available
                    # Otherwise use UUID
                    try:
                        meta = paper_data[0].get('metadata', {})
                        year = meta.get('year', 'UnknownYear').replace(" ", "_")
                        cls = meta.get('class', 'UnknownClass').replace(" ", "_")
                        sub = meta.get('subject', 'UnknownSubject').replace(" ", "_")
                        
                        # Dynamic Folder Structure
                        save_dir = Path(OUTPUT_FOLDER) / year / cls / sub
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        filename = f"{uuid.uuid4()}.json"
                        final_path = save_dir / filename
                    except:
                        # Fallback
                        final_path = Path(OUTPUT_FOLDER) / f"{uuid.uuid4()}.json"

                    # Save to Disk
                    with open(final_path, 'w', encoding='utf-8') as f:
                        json.dump(paper_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"  Saved to {final_path}")

                except Exception as e:
                    print(f"CRITICAL ERROR on {file}: {e}")

if __name__ == "__main__":
    main()
