import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM
import gc
import os

# --- CONFIGURATION ---
PDF_IMAGE_DIR = "./output/2023/XII/Download/images" 
INTERMEDIATE_DIR = "./intermediate_text"
FINAL_JSON_DIR = "./final_json"

os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
os.makedirs(FINAL_JSON_DIR, exist_ok=True)

def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# PASS 1: THE EYE (Qwen2-VL-2B)
# ==========================================
def run_pass_1_ocr():
    print("\n>>> [PASS 1] LOADING VISION MODEL (Qwen2-VL-2B)...")
    
    # We use the 2B-Instruct model. It fits easily in 8GB VRAM.
    # If you want to push limits, try "Qwen/Qwen2-VL-7B-Instruct-AWQ"
    model_path = "Qwen/Qwen2-VL-2B-Instruct"
    
    # Load with Flash Attention if available for max memory savings
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="cuda",
            attn_implementation="flash_attention_2" 
        )
        print(">>> Flash Attention 2 Enabled.")
    except Exception:
        print(">>> Flash Attention 2 Failed. Falling back to default.")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="cuda"
        )

    processor = AutoProcessor.from_pretrained(model_path)
    
    image_files = sorted([f for f in os.listdir(PDF_IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    for img_file in image_files:
        print(f"--> Reading: {img_file}")
        image_path = os.path.join(PDF_IMAGE_DIR, img_file)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Transcribe the text on this page exactly as it appears. Do not format it as JSON. Just give me the raw text content."}
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
        ).to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=1500) # Cap at 1500 tokens to prevent loops
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Save Raw Text
        txt_filename = os.path.splitext(img_file)[0] + ".txt"
        with open(os.path.join(INTERMEDIATE_DIR, txt_filename), "w", encoding="utf-8") as f:
            f.write(output_text)
            
    # CRITICAL: Clean up
    del model
    del processor
    cleanup_gpu()
    print(">>> [PASS 1] COMPLETE. GPU RELEASED.")

# ==========================================
# PASS 2: THE BRAIN (Qwen2.5-7B-Int4)
# ==========================================
def run_pass_2_formatting():
    print("\n>>> [PASS 2] LOADING LOGIC MODEL (Qwen2.5-7B-Int4)...")
    
    model_id = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype="auto"
    )

    text_files = sorted([f for f in os.listdir(INTERMEDIATE_DIR) if f.endswith('.txt')])
    
    for txt_file in text_files:
        print(f"--> Structuring: {txt_file}")
        with open(os.path.join(INTERMEDIATE_DIR, txt_file), "r", encoding="utf-8") as f:
            raw_content = f.read()

        prompt = f"""
        You are an API that converts raw exam text into JSON.
        
        RULES:
        1. Extract the Subject Name.
        2. Extract each question into a list.
        3. Do not change the Hindi text.
        4. Output valid JSON only. No markdown ```json blocks.
        
        RAW CONTENT:
        {raw_content[:3000]} 
        
        JSON OUTPUT:
        """
        # Note: I sliced content[:3000] just in case OCR is massive, to fit context.
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=2048,
            temperature=0.1
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for output_ids, input_ids in zip(generated_ids, model_inputs.input_ids)
        ]
        json_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Save JSON
        json_filename = os.path.splitext(txt_file)[0] + ".json"
        with open(os.path.join(FINAL_JSON_DIR, json_filename), "w", encoding="utf-8") as f:
            f.write(json_output)

    print(">>> [PASS 2] COMPLETE.")

if __name__ == "__main__":
    run_pass_1_ocr() 
    run_pass_2_formatting()

