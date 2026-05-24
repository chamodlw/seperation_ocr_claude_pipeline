"""
Step 6: Evaluate Trained Model
===============================
- Loads trained model and encoded test set
- Prints per-class F1 scores
- Flags classes that need more annotation

Usage:
    python step6_evaluate.py

Target metrics:
    headline      F1 > 0.90
    article_body  F1 > 0.88
    advertisement F1 > 0.85
"""

import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report, f1_score


# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH   = './models/sinhala-layoutlmv3-final'
ENCODED_DIR  = 'annotated_data/encoded_dataset'

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

TARGET_F1 = {
    'headline':     0.90,
    'article_body': 0.88,
    'advertisement':0.85,
    'full_article': 0.88,
}
# ──────────────────────────────────────────────────────────────────────────────


def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model not found: {MODEL_PATH}")
        print("    Run step4_train.py first.")
        return

    if not os.path.exists(ENCODED_DIR):
        print(f"[!] Encoded dataset not found: {ENCODED_DIR}")
        print("    Run step3_preprocess.py first.")
        return

    print("[→] Loading model and dataset...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()

    encoded_dataset = load_from_disk(ENCODED_DIR)
    test_set = encoded_dataset['test']
    test_loader = DataLoader(test_set, batch_size=1)

    all_preds, all_labels = [], []

    print(f"[→] Running inference on {len(test_set)} test pages...")
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
            preds = outputs.logits.argmax(-1)
            for pred_row, label_row in zip(preds, batch['labels']):
                all_preds.append([
                    ID2LABEL[p.item()]
                    for p, l in zip(pred_row, label_row)
                    if l.item() != -100
                ])
                all_labels.append([
                    ID2LABEL[l.item()]
                    for l in label_row
                    if l.item() != -100
                ])

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(classification_report(all_labels, all_preds))

    overall_f1 = f1_score(all_labels, all_preds)
    print(f"Overall F1: {overall_f1:.4f}")

    # Per-class check against targets
    print("\n" + "="*60)
    print("TARGET CHECK")
    print("="*60)
    from seqeval.metrics import classification_report as cr
    report_dict = cr(all_labels, all_preds, output_dict=True)

    all_passed = True
    for cls, target in TARGET_F1.items():
        key = cls
        score = report_dict.get(key, {}).get('f1-score', 0.0)
        status = '✓ PASS' if score >= target else '✗ FAIL'
        if score < target:
            all_passed = False
        print(f"  {cls:<20} F1={score:.3f}  target={target:.2f}  {status}")

    print()
    if all_passed:
        print("[✓] All targets met! Model is ready for production.")
    else:
        print("[!] Some classes below target.")
        print("    Fix: annotate 50 more pages focused on failing classes,")
        print("    re-run step2 → step3 → step4, then evaluate again.")


if __name__ == '__main__':
    evaluate()
