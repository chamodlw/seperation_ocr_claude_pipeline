"""
Step 2: Build LayoutLMv3 Dataset from Label Studio Annotations
==============================================================
- Runs Sinhala OCR on each annotated image
- Assigns BIO labels from Label Studio bounding boxes
- Saves dataset to annotated_data/sinhala_dataset.json

Usage:
    1. Annotate images in Label Studio (see README.md for setup)
    2. Export annotations as JSON-MIN from Label Studio
    3. Run: python step2_build_dataset.py
"""

import json
import os
import pytesseract
from PIL import Image

# Disable PIL decompression bomb limit — newspaper scans can exceed 89MP
Image.MAX_IMAGE_PIXELS = None


# ─── Label mapping ────────────────────────────────────────────────────────────
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


def assign_label(cx, cy, annotations, img_w, img_h):
    """Find which annotated region a word's center falls in."""
    for ann in annotations:
        for result in ann.get('result', []):
            val = result.get('value', {})
            if 'rectanglelabels' not in val:
                continue
            rx = val['x'] / 100 * img_w
            ry = val['y'] / 100 * img_h
            rw = val['width'] / 100 * img_w
            rh = val['height'] / 100 * img_h
            if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
                return val['rectanglelabels'][0]
    return 'other'


def build_bio_labels(word_labels):
    """Convert flat labels to BIO format."""
    bio = []
    prev = None
    for label in word_labels:
        if label == 'other' or label == 'O':
            bio.append('O')
            prev = None
        elif label != prev:
            bio.append(f'B-{label}')
            prev = label
        else:
            bio.append(f'I-{label}')
    return bio


def build_dataset(images_dir='output_images',
                  annotations_file='annotated_data/annotations.json',
                  output_file='annotated_data/sinhala_dataset.json'):

    os.makedirs('annotated_data', exist_ok=True)

    if not os.path.exists(annotations_file):
        print(f"[!] Annotations file not found: {annotations_file}")
        print("    Please export your Label Studio annotations as JSON-MIN")
        print("    and save to annotated_data/annotations.json")
        return

    with open(annotations_file, encoding='utf-8') as f:
        annotations = json.load(f)

    print(f"[✓] Loaded {len(annotations)} annotated pages")
    dataset = []

    for ann in annotations:
        img_filename = os.path.basename(ann.get('file_upload', ''))
        img_path = os.path.join(images_dir, img_filename)

        if not os.path.exists(img_path):
            # Check if there is a Label Studio hash prefix (e.g. 8 hex chars + '-')
            parts = img_filename.split('-', 1)
            if len(parts) > 1 and len(parts[0]) == 8:
                fallback_filename = parts[1]
                fallback_path = os.path.join(images_dir, fallback_filename)
                if os.path.exists(fallback_path):
                    img_filename = fallback_filename
                    img_path = fallback_path

        if not os.path.exists(img_path):
            print(f"  [SKIP] Image not found: {img_path}")
            continue

        print(f"  Processing: {img_filename}")
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size

        # Run Sinhala OCR
        ocr_data = pytesseract.image_to_data(
            img, lang='sin',
            output_type=pytesseract.Output.DICT
        )

        words, boxes, raw_labels = [], [], []
        for i, word in enumerate(ocr_data['text']):
            if not word.strip():
                continue
            conf = int(ocr_data['conf'][i])
            if conf < 20:
                continue
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]

            norm_box = [
                int(x / img_w * 1000),
                int(y / img_h * 1000),
                int((x + w) / img_w * 1000),
                int((y + h) / img_h * 1000),
            ]

            cx, cy = x + w // 2, y + h // 2
            label = assign_label(cx, cy, ann.get('annotations', []), img_w, img_h)

            words.append(word)
            boxes.append(norm_box)
            raw_labels.append(label)

        bio_labels = build_bio_labels(raw_labels)

        dataset.append({
            'image': img_path,
            'words': words,
            'boxes': boxes,
            'labels': bio_labels,
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] Dataset saved → {output_file}")
    print(f"    Total pages processed: {len(dataset)}")


if __name__ == '__main__':
    build_dataset()
