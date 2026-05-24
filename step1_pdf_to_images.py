"""
Step 1: Convert Sinhala Newspaper PDFs to High-Resolution Images
================================================================
Input  : PDF files in input_pdfs/
Output : PNG images in output_images/
"""

from pdf2image import convert_from_path
import os
import sys

def convert_pdfs(pdf_dir='input_pdfs', output_dir='output_images', dpi=300):
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"[!] No PDF files found in '{pdf_dir}/'")
        print("    Please put your newspaper PDFs in the input_pdfs/ folder.")
        return

    print(f"[✓] Found {len(pdf_files)} PDF file(s) to convert")
    total_pages = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"\n→ Converting: {pdf_file}")
        try:
            pages = convert_from_path(pdf_path, dpi=dpi, fmt='PNG')
            base = os.path.splitext(pdf_file)[0]
            for i, page in enumerate(pages):
                out_path = os.path.join(output_dir, f"{base}_page_{i+1:03d}.png")
                page.save(out_path)
                print(f"  Saved page {i+1}: {out_path}")
                total_pages += 1
        except Exception as e:
            print(f"  [ERROR] Failed to convert {pdf_file}: {e}")

    print(f"\n[✓] Done! Converted {total_pages} page(s) → saved in '{output_dir}/'")


if __name__ == '__main__':
    convert_pdfs()
