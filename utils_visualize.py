"""
Visualization Utility
======================
Draw colored bounding boxes on newspaper images to verify predictions.

Usage:
    python utils_visualize.py --image output_images/page_001.png
"""

import cv2
import numpy as np
import argparse
import os
import pytesseract
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

MODEL_PATH = './models/sinhala-layoutlmv3-final'

COLORS = {
    'headline':     (127, 119, 221),   # purple
    'article_body': (29,  158, 117),   # teal
    'advertisement':(216,  90,  48),   # coral
    'image_caption':(212,  83, 126),   # pink
    'other':        (136, 135, 128),   # gray
    'full_article': (0,  196, 245),    # yellow/gold (#F5C400)
}

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


def visualize(image_path, output_path=None):
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model not found: {MODEL_PATH}. Run step4_train.py first.")
        return

    processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()

    img_pil = Image.open(image_path).convert('RGB')
    img_w, img_h = img_pil.size

    ocr = pytesseract.image_to_data(
        img_pil, lang='sin',
        output_type=pytesseract.Output.DICT
    )

    words, norm_boxes, positions = [], [], []
    for i, word in enumerate(ocr['text']):
        if not word.strip() or int(ocr['conf'][i]) < 30:
            continue
        x, y, w, h = (ocr['left'][i], ocr['top'][i],
                      ocr['width'][i], ocr['height'][i])
        norm_boxes.append([
            int(x/img_w*1000), int(y/img_h*1000),
            int((x+w)/img_w*1000), int((y+h)/img_h*1000)
        ])
        words.append(word)
        positions.append((x, y, w, h))

    if not words:
        print("[!] No words detected by OCR.")
        return

    encoding = processor(
        img_pil, words, boxes=norm_boxes,
        truncation=True, padding='max_length',
        max_length=512, return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**encoding)
    preds = outputs.logits.argmax(-1).squeeze().tolist()
    labels = [ID2LABEL.get(p, 'O') for p in preds[:len(words)]]

    # Draw on image
    img_cv = cv2.imread(image_path)
    for label, (x, y, w, h) in zip(labels, positions):
        if label == 'O':
            continue
        cls = label[2:] if label.startswith(('B-', 'I-')) else label
        color = COLORS.get(cls, (128, 128, 128))
        cv2.rectangle(img_cv, (x, y), (x+w, y+h), color, 2)

    # Legend
    legend_y = 30
    for cls, color in COLORS.items():
        cv2.rectangle(img_cv, (10, legend_y-15), (30, legend_y), color, -1)
        cv2.putText(img_cv, cls, (35, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0,0), 1)
        legend_y += 25

    if output_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"output/{base}_visualized.png"

    os.makedirs('output', exist_ok=True)
    cv2.imwrite(output_path, img_cv)
    print(f"[✓] Visualization saved → {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to newspaper image')
    parser.add_argument('--output', default=None, help='Output path for visualization')
    args = parser.parse_args()
    visualize(args.image, args.output)
