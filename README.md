# Sinhala Newspaper Article Separation Pipeline
### LayoutLMv3 Training Method

---

## What This Does

Automatically separates news articles and advertisements from Sinhala newspaper PDFs using a fine-tuned LayoutLMv3 model. Each detected region is saved as a cropped image.

```
PDF  →  Images  →  Annotate (once)  →  Train Model  →  Auto-separate new newspapers
```

---

## Output Structure

```
output/
└── 2024-11-11/
    ├── articles/
    │   ├── page_001/
    │   │   ├── article_1.png
    │   │   ├── article_2.png
    │   │   └── article_3.png
    │   └── page_002/
    │       └── article_1.png
    └── ads/
        ├── page_001/
        │   └── ad_1.png
        └── page_002/
            └── ad_1.png
```

---

## Setup

```cmd
conda create -n layoutlm python=3.9 -y
conda activate layoutlm
pip install torch torchvision
pip install transformers==4.38.0 datasets seqeval timm
pip install pytesseract opencv-python pdf2image label-studio
```

Also install:
- **Tesseract OCR** with Sinhala language pack (`sin`)
- **Poppler** for Windows — add `C:\poppler\Library\bin` to PATH

---

## Steps

> Always use **CMD** (not PowerShell) with `conda activate layoutlm`

### 1 — Convert PDFs to Images
```cmd
python step1_pdf_to_images.py
```
Put PDFs in `input_pdfs/` first. Output saved to `output_images/` at 300 DPI.

---

### 2 — Annotate with Label Studio
```cmd
label-studio start --port 8080
```
Open **http://localhost:8080** then:
1. Create project → **Object Detection with Bounding Boxes**
2. Settings → Labeling Interface → Code → paste:

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="full_article"  background="#F5C400"/>
    <Label value="advertisement" background="#D85A30"/>
  </RectangleLabels>
</View>
```

3. Import images from `output_images/`
4. Draw bounding boxes — **150+ pages** needed for good accuracy
5. Export as **JSON** → save to `annotated_data/annotations.json`

**Annotation tips:**

| Label | Color | Draw around |
|-------|-------|-------------|
| `full_article` | 🟡 Yellow | Each complete news article (headline + body together) |
| `advertisement` | 🟠 Orange | Ads and promotional content |

- For L-shaped articles: draw 2 separate `full_article` boxes
- Ignore photos, page numbers, borders — leave them unannotated
- **For better accuracy on unseen newspapers**: annotate pages from at least 3 different newspapers (e.g. Lankadeepa, Dinamina, Aruna). A model trained on one paper's layout will struggle with others.

---

### 3 — Build Dataset
```cmd
python step2_build_dataset.py
```

### 4 — Preprocess
```cmd
python step3_preprocess.py
```

### 5 — Train Model
```cmd
python step4_train.py
```
> Runs overnight on CPU. Best model saved to `models/sinhala-layoutlmv3-final/`

Training uses cosine LR scheduling and label smoothing so the model generalises better to newspaper layouts it has not seen before.

### 6 — Separate Articles from New Newspapers
```cmd
python step1_pdf_to_images.py
python step5_inference.py
```
Cropped images saved to `output/{date}/articles/` and `output/{date}/ads/`

### 7 — Evaluate Accuracy
```cmd
python step6_evaluate.py
```

| Class | Target F1 |
|-------|-----------|
| full_article | > 0.88 |
| advertisement | > 0.85 |

If a class is below target: annotate 50 more pages focused on that class (from different newspapers), then re-run steps 3 → 4 → 7.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `conda not recognized` | Use CMD not PowerShell |
| Poppler error | Install poppler, add `C:\poppler\Library\bin` to PATH |
| `sin` not in tesseract | Reinstall Tesseract with Sinhala language pack checked |
| Label Studio image error | Run `set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` before starting |
| Low accuracy on new papers | Annotate pages from more newspapers — layout diversity is the key factor |
| No output after inference | Run step4_train.py first — model must be trained |
