import argparse
import torch
import gc
from pathlib import Path
from pdf2image import convert_from_path
from transformers import NougatProcessor, VisionEncoderDecoderModel
from PIL import Image

# --- CONFIGURATION ---
# Nougat was trained on 300 DPI images, so we use that for best results.
OCR_DPI = 300

# VRAM safety setting. If you have a lot of VRAM, you can increase this.
MAX_IMAGE_DIMENSION = 1600

def cleanup_gpu():
    """Frees up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def run_nougat_ocr(pdf_path: Path, output_dir: Path):
    """
    Runs the Nougat model to perform OCR on a single PDF and saves the output.
    This model is specialized for academic documents and extracts English text.
    """
    if not pdf_path.exists():
        print(f"!! Error: PDF file not found at {pdf_path}")
        return

    print(f"\n{'='*50}\n>>> Running Nougat OCR on: {pdf_path.name}\n{'='*50}")

    # 1. Load Model and Processor
    print("  - Loading Nougat model ('unstructured-io/nougat-base')...")
    try:
        processor = NougatProcessor.from_pretrained("unstructured-io/nougat-base")
        # For better performance on capable GPUs, use bfloat16
        model = VisionEncoderDecoderModel.from_pretrained(
            "unstructured-io/nougat-base",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa" # Use Flash Attention 2 if available
        )
    except Exception as e:
        print(f"!! Error loading model: {e}")
        print("   Please ensure you have 'transformers', 'torch', and 'accelerate' installed.")
        return

    output_file = output_dir / f"{pdf_path.stem}_nougat_ocr.txt"
    full_text_content = ""

    try:
        # 2. Convert PDF to images
        print(f"  - Converting PDF to images at {OCR_DPI} DPI...")
        images = convert_from_path(pdf_path, dpi=OCR_DPI)

        # 3. Process each page
        for i, image in enumerate(images):
            print(f"    - Processing page {i+1}/{len(images)}...")
            if max(image.size) > MAX_IMAGE_DIMENSION:
                image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)

            pixel_values = processor(image, return_tensors="pt").pixel_values

            with torch.no_grad():
                outputs = model.generate(
                    pixel_values.to(model.device, dtype=torch.bfloat16),
                    min_length=1,
                    max_new_tokens=4096,
                )

            # The model outputs markdown-like text, which is great for structure
            page_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            full_text_content += f"\n\n--- PAGE {i+1} ---\n\n{page_text}"
            del pixel_values, outputs
            cleanup_gpu()

        # 4. Save the final output
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_text_content)
        print(f"\n  [Success] OCR complete. Output saved to: {output_file}")

    except Exception as e:
        print(f"\n  !! An error occurred during processing: {e}")
    finally:
        # 5. Unload model from memory
        del model, processor
        cleanup_gpu()
        print("  - Nougat model unloaded.")

def main():
    parser = argparse.ArgumentParser(description="Extract English text from a PDF using the Nougat model.")
    parser.add_argument("pdf_path", type=str, help="The absolute or relative path to the PDF file.")
    args = parser.parse_args()

    run_nougat_ocr(Path(args.pdf_path), Path.cwd())

if __name__ == "__main__":
    main()