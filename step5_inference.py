"""
Step 5: Run Inference — Detect & Extract Articles from New Newspaper Pages
==========================================================================
- Loads trained model from models/sinhala-layoutlmv3-final/
- Runs Sinhala OCR + LayoutLMv3 on each image
- Groups tokens into articles (headline + body)
- Saves results to output/extracted_articles.json

Usage:
    python step5_inference.py
    python step5_inference.py --image output_images/lankadeepa_page_001.png
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
    'B-headline': 1,    'I-headline': 2,
    'B-article_body': 3,'I-article_body': 4,
    'B-advertisement': 5,'I-advertisement': 6,
    'B-image_caption': 7,'I-image_caption': 8,
    'B-other': 9,       'I-other': 10,
    'B-full_article': 11,'I-full_article': 12,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
# ──────────────────────────────────────────────────────────────────────────────


def load_model(model_path):
    """Load trained LayoutLMv3 model and processor."""
    # Fall back to base model if trained model not found yet
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
    """Run Tesseract OCR and return words + normalized boxes + raw positions."""
    img_w, img_h = img.size
    ocr = pytesseract.image_to_data(
        img, lang=lang,
        output_type=pytesseract.Output.DICT
    )

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

        x = ocr['left'][i]
        y = ocr['top'][i]
        w = ocr['width'][i]
        h = ocr['height'][i]

        norm_box = [
            int(x / img_w * 1000),
            int(y / img_h * 1000),
            int((x + w) / img_w * 1000),
            int((y + h) / img_h * 1000),
        ]
        words.append(word)
        norm_boxes.append(norm_box)
        positions.append((x, y, w, h))

    return words, norm_boxes, positions


def predict_page(image_path, processor, model):
    """Run model inference on a single page image."""
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


def extract_articles(words, token_labels, positions):
    """Group predicted tokens into article regions and match to headlines."""
    regions = []
    current = None

    for word, label, pos in zip(words, token_labels, positions):
        if label.startswith('B-'):
            if current:
                regions.append(current)
            current = {
                'type': label[2:],
                'words': [word],
                'pos': [pos],
            }
        elif label.startswith('I-') and current:
            current['words'].append(word)
            current['pos'].append(pos)
        else:
            if current:
                regions.append(current)
                current = None

    if current:
        regions.append(current)

    headlines = [r for r in regions if r['type'] == 'headline']
    bodies    = [r for r in regions if r['type'] == 'article_body']
    full_articles = [r for r in regions if r['type'] == 'full_article']

    articles = []

    # 1. Extract full_article regions directly
    for fa in full_articles:
        news_text = ' '.join(fa['words']).strip()
        if news_text:
            articles.append({
                'news': news_text,
                'pos': list(fa['pos']),
            })

    # 2. Extract legacy article_body regions matched to headlines
    for body in bodies:
        body_top = min(p[1] for p in body['pos'])

        # Find closest headline above this body
        candidates = [
            h for h in headlines
            if min(p[1] for p in h['pos']) < body_top
        ]
        all_positions = list(body['pos'])
        if candidates:
            headline = max(
                candidates,
                key=lambda h: min(p[1] for p in h['pos'])
            )
            headline_text = ' '.join(headline['words'])
            all_positions.extend(headline['pos'])
        else:
            headline_text = ''

        body_text = ' '.join(body['words'])
        news_text = f"{headline_text} {body_text}".strip() if headline_text else body_text
        if news_text:
            articles.append({
                'news': news_text,
                'pos': all_positions,
            })

    return articles


def crop_and_save_article(image_path, pos_list, output_path, padding=15):
    """Crop article from original image using its word positions and save it."""
    if not pos_list:
        return None

    # Calculate overall bounding box enclosing all word boxes in pos_list
    # Each pos is (x, y, w, h)
    xmin = min(p[0] for p in pos_list)
    ymin = min(p[1] for p in pos_list)
    xmax = max(p[0] + p[2] for p in pos_list)
    ymax = max(p[1] + p[3] for p in pos_list)

    try:
        img = Image.open(image_path)
        img_w, img_h = img.size

        # Apply padding and clamp to image dimensions
        xmin = max(0, xmin - padding)
        ymin = max(0, ymin - padding)
        xmax = min(img_w, xmax + padding)
        ymax = min(img_h, ymax + padding)

        # Crop and save
        cropped_img = img.crop((xmin, ymin, xmax, ymax))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cropped_img.save(output_path)
        return output_path
    except Exception as e:
        print(f"  [ERROR] Failed to crop or save article: {e}")
        return None


def process_image(image_path, processor, model):
    """Full pipeline for one image."""
    words, labels, positions = predict_page(image_path, processor, model)
    if not words:
        return []
    articles = extract_articles(words, labels, positions)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    for idx, art in enumerate(articles):
        output_img_path = f"output/separated_articles/{base_name}_article_{idx+1}.png"
        saved_path = crop_and_save_article(image_path, art.get('pos'), output_img_path)
        if saved_path:
            # Save path as relative/standardized format
            art['image_path'] = saved_path.replace('\\', '/')
        
        # Remove 'pos' to avoid JSON clutter
        if 'pos' in art:
            del art['pos']

    return articles


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

    for img_path in image_paths:
        print(f"→ {os.path.basename(img_path)}")
        articles = process_image(img_path, processor, model)
        all_results[os.path.basename(img_path)] = articles
        total_articles += len(articles)
        print(f"  Found {len(articles)} article(s)")
        for i, a in enumerate(articles):
            news_preview = a['news'][:60] if a.get('news') else '(no news text)'
            print(f"    [{i+1}] {news_preview}...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] Done! Total articles extracted: {total_articles}")
    print(f"    Results saved → {OUTPUT_FILE}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None,
                        help='Process a single image file')
    args = parser.parse_args()
    main(single_image=args.image)
