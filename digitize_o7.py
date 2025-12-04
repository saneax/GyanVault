import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import AutoModelForCausalLM
import gc
import os
import json

# --- CONFIGURATION ---
PDF_IMAGE_DIR = "./output/2023/XII/Download/images" # Where you extracted PDF pages
INTERMEDIATE_DIR = "./intermediate_text"
FINAL_JSON_DIR = "./final_json"

os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
os.makedirs(FINAL_JSON_DIR, exist_ok=True)

def cleanup_gpu():
    """Force garbage collection to free VRAM for the next model"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ==========================================
# PASS 1: THE EYE (Vision -> Raw Text)
# ==========================================
def run_pass_1_ocr():
    print(">>> LOADING PASS 1 MODEL (Vision)...")
    # Using Qwen2-VL-2B for speed and low VRAM, or use 7B-Int4 if you need more detail
    model_path = "Qwen/Qwen2-VL-2B-Instruct" 
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    image_files = sorted([f for f in os.listdir(PDF_IMAGE_DIR) if f.endswith(('.png', '.jpg'))])
    
    for img_file in image_files:
        print(f"Processing OCR for: {img_file}")
        image_path = os.path.join(PDF_IMAGE_DIR, img_file)
        
        # Simple Prompt: Just read, don't think about JSON.
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Extract all text from this page line by line. Preserve the layout as much as possible. Do not interpret, just transcribe."}
                ],
            }
        ]
        
        # Inference
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            text=[text_prompt],
            images=[Image.open(image_path)],
            padding=True,
            return_tensors="pt"
        ).to("cuda")
        
        output_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids = [output_ids[len(inputs.input_ids):] for output_ids, inputs.input_ids in zip(output_ids, inputs.input_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        # Save Raw Text
        txt_filename = os.path.splitext(img_file)[0] + ".txt"
        with open(os.path.join(INTERMEDIATE_DIR, txt_filename), "w", encoding="utf-8") as f:
            f.write(output_text)
            
    # CRITICAL: Delete model and free memory
    del model
    del processor
    cleanup_gpu()
    print(">>> PASS 1 COMPLETE. GPU CLEARED.")

# ==========================================
# PASS 2: THE BRAIN (Raw Text -> JSON)
# ==========================================
def run_pass_2_formatting():
    print(">>> LOADING PASS 2 MODEL (Logic)...")
    # Using a strong text model for instruction following
    model_id = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype="auto"
    )

    text_files = sorted([f for f in os.listdir(INTERMEDIATE_DIR) if f.endswith('.txt')])
    
    for txt_file in text_files:
        print(f"Formatting JSON for: {txt_file}")
        with open(os.path.join(INTERMEDIATE_DIR, txt_file), "r", encoding="utf-8") as f:
            raw_content = f.read()

        # The Prompt: Strict JSON enforcement
        prompt = f"""
        You are a data formatting assistant. 
        Convert the following raw text from an exam paper into a strict JSON format.
        
        JSON Schema required:
        {{
            "subject": "string",
            "questions": [
                {{ "q_no": "string", "text": "string", "marks": "string" }}
            ]
        }}

        RAW TEXT:
        {raw_content}
        
        OUTPUT JSON ONLY:
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=2048,
            temperature=0.1 # Low temp for deterministic structure
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for output_ids, input_ids in zip(generated_ids, model_inputs.input_ids)
        ]
        json_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Save JSON
        json_filename = os.path.splitext(txt_file)[0] + ".json"
        with open(os.path.join(FINAL_JSON_DIR, json_filename), "w", encoding="utf-8") as f:
            f.write(json_output)

    print(">>> PASS 2 COMPLETE.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # run_pass_1_ocr() 
    run_pass_2_formatting() # Uncomment logic as needed to run steps individually or together

