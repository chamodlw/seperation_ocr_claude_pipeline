"""
Step 5: Extract & Separate Articles from New Newspaper Pages
============================================================
- Loads trained model from models/sinhala-layoutlmv3-final/
- Runs Sinhala OCR + LayoutLMv3 on each page image
- Crops each detected region and saves it

Output structure:
    output/{newspaper}/articles/{page}/article_N.png
    output/{newspaper}/ads/{page}/ad_N.png

Usage:
    python step5_inference.py
    python step5_inference.py --image output_images/2024-11-11_page_001.png
"""

import os
import json
import argparse
import torch
import pytesseract
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification


# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = './models/sinhala-layoutlmv3-final'
IMAGE_DIR   = 'output_images'
OUTPUT_FILE = 'output/extracted_articles.json'
OCR_LANG    = 'sin'
OCR_CONF_THRESHOLD = 30

LABEL2ID = {
    'O': 0,
    'B-full_article': 1,  'I-full_article': 2,
    'B-advertisement': 3, 'I-advertisement': 4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
# ──────────────────────────────────────────────────────────────────────────────


def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"[!] Trained model not found at '{model_path}'")
        print("    Falling back to base model (untrained — low accuracy)")
        print("    Run step4_train.py first for good results.\n")
        model_path = 'microsoft/layoutlmv3-base'

    print(f"[→] Loading model from: {model_path}")
    processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    return processor, model


def run_ocr(img, lang='sin', conf_threshold=30):
    img_w, img_h = img.size
    ocr = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

    words, norm_boxes, positions = [], [], []
    for i, word in enumerate(ocr['text']):
        if not word.strip():
            continue
        try:
            conf = int(ocr['conf'][i])
        except (ValueError, TypeError):
            continue
        if conf < conf_threshold:
            continue

        x, y, w, h = ocr['left'][i], ocr['top'][i], ocr['width'][i], ocr['height'][i]
        norm_boxes.append([
            int(x / img_w * 1000),
            int(y / img_h * 1000),
            int((x + w) / img_w * 1000),
            int((y + h) / img_h * 1000),
        ])
        words.append(word)
        positions.append((x, y, w, h))

    return words, norm_boxes, positions


def predict_page(image_path, processor, model):
    img = Image.open(image_path).convert('RGB')
    words, norm_boxes, positions = run_ocr(img)

    if not words:
        print(f"  [!] No words detected by OCR in: {image_path}")
        return [], [], []

    encoding = processor(
        img, words,
        boxes=norm_boxes,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt',
    )

    with torch.no_grad():
        outputs = model(**encoding)

    preds = outputs.logits.argmax(-1).squeeze().tolist()
    if isinstance(preds, int):
        preds = [preds]

    token_labels = [ID2LABEL.get(p, 'O') for p in preds[:len(words)]]
    return words, token_labels, positions


def extract_regions(words, token_labels, positions):
    """Group BIO tokens into contiguous regions. Returns (articles, ads)."""
    raw = []
    current = None

    for word, label, pos in zip(words, token_labels, positions):
        if label.startswith('B-'):
            if current:
                raw.append(current)
            current = {'type': label[2:], 'words': [word], 'pos': [pos]}
        elif label.startswith('I-') and current:
            current['words'].append(word)
            current['pos'].append(pos)
        else:
            if current:
                raw.append(current)
                current = None

    if current:
        raw.append(current)

    articles = [r for r in raw if r['type'] == 'full_article']
    ads      = [r for r in raw if r['type'] == 'advertisement']
    return articles, ads


def crop_and_save(image_path, pos_list, output_path, padding=15):
    if not pos_list:
        return None
    xmin = min(p[0] for p in pos_list)
    ymin = min(p[1] for p in pos_list)
    xmax = max(p[0] + p[2] for p in pos_list)
    ymax = max(p[1] + p[3] for p in pos_list)

    try:
        img = Image.open(image_path)
        img_w, img_h = img.size
        xmin = max(0,     xmin - padding)
        ymin = max(0,     ymin - padding)
        xmax = min(img_w, xmax + padding)
        ymax = min(img_h, ymax + padding)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.crop((xmin, ymin, xmax, ymax)).save(output_path)
        return output_path
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def parse_image_name(image_path):
    """'2024-11-11_page_001.png' → ('2024-11-11', 'page_001')"""
    base = os.path.splitext(os.path.basename(image_path))[0]
    if '_page_' in base:
        newspaper, page_num = base.rsplit('_page_', 1)
        return newspaper, f"page_{page_num}"
    return base, 'page_001'


def process_image(image_path, processor, model):
    words, labels, positions = predict_page(image_path, processor, model)
    if not words:
        return [], []

    raw_articles, raw_ads = extract_regions(words, labels, positions)
    newspaper, page = parse_image_name(image_path)

    articles = []
    for idx, art in enumerate(raw_articles):
        out_path = f"output/{newspaper}/articles/{page}/article_{idx+1}.png"
        saved = crop_and_save(image_path, art['pos'], out_path)
        entry = {'image_path': saved.replace('\\', '/') if saved else None}
        articles.append(entry)

    ads = []
    for idx, ad in enumerate(raw_ads):
        out_path = f"output/{newspaper}/ads/{page}/ad_{idx+1}.png"
        saved = crop_and_save(image_path, ad['pos'], out_path)
        entry = {'image_path': saved.replace('\\', '/') if saved else None}
        ads.append(entry)

    return articles, ads


def main(single_image=None):
    os.makedirs('output', exist_ok=True)
    processor, model = load_model(MODEL_PATH)

    if single_image:
        image_paths = [single_image]
    else:
        if not os.path.exists(IMAGE_DIR):
            print(f"[!] Image directory not found: {IMAGE_DIR}")
            print("    Run step1_pdf_to_images.py first.")
            return
        image_paths = [
            os.path.join(IMAGE_DIR, f)
            for f in sorted(os.listdir(IMAGE_DIR))
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    if not image_paths:
        print("[!] No images found to process.")
        return

    print(f"[✓] Processing {len(image_paths)} image(s)...\n")
    all_results = {}
    total_articles = 0
    total_ads = 0

    for img_path in image_paths:
        print(f"→ {os.path.basename(img_path)}")
        newspaper, page = parse_image_name(img_path)
        articles, ads = process_image(img_path, processor, model)

        all_results.setdefault(newspaper, {})[page] = {
            'articles': articles,
            'ads':      ads,
        }
        total_articles += len(articles)
        total_ads      += len(ads)
        print(f"  {len(articles)} article(s), {len(ads)} ad(s)")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] Done!")
    print(f"    Articles : {total_articles}")
    print(f"    Ads      : {total_ads}")
    print(f"    Results  → {OUTPUT_FILE}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None,
                        help='Process a single image file')
    args = parser.parse_args()
    main(single_image=args.image)
