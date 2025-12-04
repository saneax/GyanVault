import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from pdf2image import convert_from_path
import gc
import os
import shutil

# --- CONFIGURATION ---
# The folder where your PDF lives
SOURCE_PDF_DIR = "./output/2023/XII/Download" 

# Where we will dump the images automatically
GENERATED_IMAGES_DIR = "./output/2023/XII/Download/images"
INTERMEDIATE_DIR = "./intermediate_text"
FINAL_JSON_DIR = "./final_json"

for d in [GENERATED_IMAGES_DIR, INTERMEDIATE_DIR, FINAL_JSON_DIR]:
    os.makedirs(d, exist_ok=True)

def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# PRE-PASS: PDF -> IMAGES
# ==========================================
def convert_pdfs():
    print(f"\n>>> [PRE-PASS] Converting PDFs in {SOURCE_PDF_DIR}...")
    
    # Find all PDFs
    pdf_files = [f for f in os.listdir(SOURCE_PDF_DIR) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("!!! NO PDFs FOUND. Check your path.")
        return

    for pdf_file in pdf_files:
        full_path = os.path.join(SOURCE_PDF_DIR, pdf_file)
        print(f"--> Converting: {pdf_file}")
        
        try:
            # Convert to images (300 DPI is good for OCR)
            images = convert_from_path(full_path, dpi=300)
            
            base_name = os.path.splitext(pdf_file)[0]
            for i, image in enumerate(images):
                # Save as formatted filename: "BusinessStudies_page01.png"
                image_name = f"{base_name}_page{i+1:02d}.png"
                save_path = os.path.join(GENERATED_IMAGES_DIR, image_name)
                image.save(save_path, "PNG")
                print(f"    Saved: {image_name}")
                
        except Exception as e:
            print(f"!!! Error converting {pdf_file}: {e}")

# ==========================================
# PASS 1: THE EYE (Qwen2-VL-2B)
# ==========================================
def run_pass_1_ocr():
    print("\n>>> [PASS 1] LOADING VISION MODEL (Qwen2-VL-2B)...")
    
    model_path = "Qwen/Qwen2-VL-2B-Instruct"
    
    # Added bfloat16 to fix the warning and improve speed
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda",
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Process all images we just generated
    image_files = sorted([f for f in os.listdir(GENERATED_IMAGES_DIR) if f.endswith('.png')])
    
    if not image_files:
        print("!!! No images found to process.")
        return

    for img_file in image_files:
        print(f"--> Reading: {img_file}")
        image_path = os.path.join(GENERATED_IMAGES_DIR, img_file)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Transcribe the text on this page exactly as it appears. Include any Hindi text accurately. Do not format it as JSON yet."}
                ],
            }
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

        generated_ids = model.generate(**inputs, max_new_tokens=1500)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        txt_filename = os.path.splitext(img_file)[0] + ".txt"
        with open(os.path.join(INTERMEDIATE_DIR, txt_filename), "w", encoding="utf-8") as f:
            f.write(output_text)
            
    del model
    del processor
    cleanup_gpu()
    print(">>> [PASS 1] COMPLETE.")

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
        1. Extract the Questions, Marks, and Question Numbers.
        2. Preserve Hindi text exactly.
        3. Output valid JSON only.
        
        RAW CONTENT:
        {raw_content[:3500]} 
        
        JSON OUTPUT:
        """
        
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
        
        json_filename = os.path.splitext(txt_file)[0] + ".json"
        with open(os.path.join(FINAL_JSON_DIR, json_filename), "w", encoding="utf-8") as f:
            f.write(json_output)

    print(">>> [PASS 2] COMPLETE.")

if __name__ == "__main__":
    convert_pdfs()       # Step 1: Create Images
    run_pass_1_ocr()     # Step 2: Read Text
    run_pass_2_formatting() # Step 3: Format JSON

