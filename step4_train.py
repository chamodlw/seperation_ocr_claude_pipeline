"""
Step 4: Fine-tune LayoutLMv3 on Sinhala Newspaper Dataset
==========================================================
- Loads encoded dataset from disk
- Fine-tunes microsoft/layoutlmv3-base
- Saves best model to models/sinhala-layoutlmv3-final/

Usage:
    python step4_train.py

Notes:
    - CPU training: expect 1–2 hours per epoch for 300 pages
    - Reduce num_train_epochs if testing
    - Increase gradient_accumulation_steps if memory is low
"""

import os
import numpy as np
from datasets import load_from_disk
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import f1_score, classification_report


# ─── Config ───────────────────────────────────────────────────────────────────
ENCODED_DIR  = 'annotated_data/encoded_dataset'
MODEL_NAME   = 'microsoft/layoutlmv3-base'
OUTPUT_DIR   = './models/sinhala-layoutlmv3'
FINAL_DIR    = './models/sinhala-layoutlmv3-final'

LABEL2ID = {
    'O': 0,
    'B-full_article': 1,  'I-full_article': 2,
    'B-advertisement': 3, 'I-advertisement': 4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# Training hyperparameters
# Lower LR + cosine schedule + label smoothing improve generalisation to
# unseen newspapers by preventing the model from over-fitting layout patterns
# of the specific papers used during annotation.
NUM_EPOCHS    = 15
BATCH_SIZE    = 1       # keep at 1 for CPU / low VRAM
GRAD_ACCUM    = 4       # effective batch size = 4
LEARNING_RATE = 5e-6    # lower than default — better generalisation
# ──────────────────────────────────────────────────────────────────────────────


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_preds, true_labels = [], []
    for pred_row, label_row in zip(predictions, labels):
        true_preds.append([
            ID2LABEL[pr] for pr, lb in zip(pred_row, label_row) if lb != -100
        ])
        true_labels.append([
            ID2LABEL[lb] for lb in label_row if lb != -100
        ])

    return {
        'f1': f1_score(true_labels, true_preds),
    }


def main():
    if not os.path.exists(ENCODED_DIR):
        print(f"[!] Encoded dataset not found: {ENCODED_DIR}")
        print("    Run step3_preprocess.py first.")
        return

    print("[→] Loading encoded dataset...")
    encoded_dataset = load_from_disk(ENCODED_DIR)
    print(f"    Train: {len(encoded_dataset['train'])} | "
          f"Val: {len(encoded_dataset['validation'])} | "
          f"Test: {len(encoded_dataset['test'])}")

    print("[→] Loading model...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)
    data_collator = DataCollatorForTokenClassification(
        processor.tokenizer, pad_to_multiple_of=8
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type='cosine',      # cosine decay → smoother convergence
        label_smoothing_factor=0.1,      # prevents overconfidence on unseen layouts
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_dir='./logs',
        logging_steps=20,
        fp16=False,          # set True only if using GPU with CUDA
        no_cuda=True,        # CPU training — remove this line if using GPU
        dataloader_num_workers=0,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validation'],
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n[→] Starting training...")
    print("    This will take several hours on CPU. Let it run overnight.")
    print("    Best model is saved automatically based on F1 score.\n")

    trainer.train()

    print(f"\n[→] Saving final model to {FINAL_DIR}...")
    trainer.save_model(FINAL_DIR)
    processor.save_pretrained(FINAL_DIR)

    print("\n[✓] Training complete!")
    print(f"    Model saved → {FINAL_DIR}/")

    # Final evaluation on test set
    print("\n[→] Running test set evaluation...")
    test_results = trainer.evaluate(encoded_dataset['test'])
    print("\n=== Test Results ===")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == '__main__':
    main()
