"""
Step 3: Preprocess Dataset for LayoutLMv3 Training
===================================================
- Loads annotated dataset JSON
- Splits into train / validation / test (70/15/15)
- Tokenizes with LayoutLMv3Processor
- Saves encoded HuggingFace dataset to disk

Usage:
    python step3_preprocess.py
"""

import json
import random
import os
from datasets import Dataset, DatasetDict
from transformers import LayoutLMv3Processor
from PIL import Image


# ─── Config ───────────────────────────────────────────────────────────────────
DATASET_FILE  = 'annotated_data/sinhala_dataset.json'
ENCODED_DIR   = 'annotated_data/encoded_dataset'
MODEL_NAME    = 'microsoft/layoutlmv3-base'
MAX_LENGTH    = 512
RANDOM_SEED   = 42

LABEL2ID = {
    'O': 0,
    'B-full_article': 1,  'I-full_article': 2,
    'B-advertisement': 3, 'I-advertisement': 4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
# ──────────────────────────────────────────────────────────────────────────────


def load_and_split(dataset_file):
    with open(dataset_file, encoding='utf-8') as f:
        data = json.load(f)

    random.seed(RANDOM_SEED)
    random.shuffle(data)

    n = len(data)
    train = data[:int(n * 0.70)]
    val   = data[int(n * 0.70):int(n * 0.85)]
    test  = data[int(n * 0.85):]

    print(f"[✓] Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return DatasetDict({
        'train':      Dataset.from_list(train),
        'validation': Dataset.from_list(val),
        'test':       Dataset.from_list(test),
    })


def preprocess_fn(examples, processor):
    images = [Image.open(p).convert('RGB') for p in examples['image']]

    # Convert string labels to IDs
    word_labels_ids = [
        [LABEL2ID.get(lbl, 0) for lbl in labels]
        for labels in examples['labels']
    ]

    encoding = processor(
        images,
        examples['words'],
        boxes=examples['boxes'],
        word_labels=word_labels_ids,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt',
    )
    return encoding


def main():
    if not os.path.exists(DATASET_FILE):
        print(f"[!] Dataset not found: {DATASET_FILE}")
        print("    Run step2_build_dataset.py first.")
        return

    print("[→] Loading processor...")
    processor = LayoutLMv3Processor.from_pretrained(
        MODEL_NAME, apply_ocr=False
    )

    print("[→] Loading and splitting dataset...")
    dataset = load_and_split(DATASET_FILE)

    print("[→] Encoding dataset (this may take a few minutes)...")
    encoded = dataset.map(
        lambda ex: preprocess_fn(ex, processor),
        batched=True,
        batch_size=4,
        remove_columns=dataset['train'].column_names,
    )
    encoded.set_format('torch')

    os.makedirs(ENCODED_DIR, exist_ok=True)
    encoded.save_to_disk(ENCODED_DIR)
    print(f"[✓] Encoded dataset saved → {ENCODED_DIR}/")


if __name__ == '__main__':
    main()
