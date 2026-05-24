# Sinhala Newspaper Article Separation Pipeline
## Using LayoutLMv3 + Tesseract OCR

---

## What This Does

```
PDF Newspaper  →  Images  →  Layout Detection  →  Article Extraction  →  Sinhala Text JSON
```

Takes Sinhala newspaper PDFs (Lankadeepa, Dinamina, Aruna, Diwaina, Mawbima),
separates individual news articles, and outputs structured JSON with:
- Article headline (Sinhala)
- Article body text (Sinhala)

---

## Folder Structure

```
sinhala_newspaper_pipeline/
│
├── input_pdfs/              ← PUT YOUR NEWSPAPER PDFs HERE
├── output_images/           ← Generated page images (auto-created)
├── annotated_data/          ← Label Studio exports + encoded dataset
├── models/                  ← Trained model saved here
├── output/                  ← Final extracted articles JSON
│
├── step1_pdf_to_images.py   ← Convert PDFs → PNG images
├── step2_build_dataset.py   ← Build training dataset from annotations
├── step3_preprocess.py      ← Tokenize + encode for LayoutLMv3
├── step4_train.py           ← Fine-tune the model
├── step5_inference.py       ← Run on new newspapers → extract articles
├── step6_evaluate.py        ← Measure F1 accuracy
│
├── utils_text_clean.py      ← Sinhala text cleaning helpers
├── utils_visualize.py       ← Draw colored boxes on images
└── requirements.txt
```

---

## Prerequisites

- Python 3.9 (in conda env `layoutlm`)
- Tesseract with Sinhala language pack installed
- All pip packages installed (see requirements.txt)

---

## Step-by-Step Guide

### STEP 1 — Put your PDFs in the input folder

Copy your newspaper PDF files into the `input_pdfs/` folder.

```
input_pdfs/
  lankadeepa_2024_01_15.pdf
  dinamina_2024_01_15.pdf
  ...
```

### STEP 2 — Convert PDFs to Images

```cmd
python step1_pdf_to_images.py
```

Output: PNG images in `output_images/` at 300 DPI.

---

### STEP 3 — Annotate Pages with Label Studio

You need to label 300–500 pages to train the model.

**Start Label Studio:**
```cmd
pip install label-studio
label-studio start --port 8080
```

Open http://localhost:8080 and:
1. Create new project → Image Segmentation
2. Paste this XML in the Label Config editor:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="headline"      background="#7F77DD"/>
    <Label value="article_body"  background="#1D9E75"/>
    <Label value="advertisement" background="#D85A30"/>
    <Label value="image_caption" background="#D4537E"/>
    <Label value="other"         background="#888780"/>
  </RectangleLabels>
</View>
```

3. Import images from `output_images/`
4. Draw bounding boxes around each region
5. **Export** as **JSON-MIN** → save as `annotated_data/annotations.json`

**Annotation tips:**
- Spend 10–15 min per page
- Label at least 300 pages for usable accuracy
- Focus on headline/body boundary — that's the hardest part

---

### STEP 4 — Build Training Dataset

```cmd
python step2_build_dataset.py
```

Runs Sinhala OCR on each annotated image and assigns BIO labels.
Output: `annotated_data/sinhala_dataset.json`

---

### STEP 5 — Preprocess for LayoutLMv3

```cmd
python step3_preprocess.py
```

Tokenizes and encodes the dataset.
Output: `annotated_data/encoded_dataset/`

---

### STEP 6 — Train the Model

```cmd
python step4_train.py
```

⚠️ **This takes a long time on CPU — let it run overnight.**
- ~1–2 hours per epoch × 15 epochs on CPU
- The best model is automatically saved to `models/sinhala-layoutlmv3-final/`

---

### STEP 7 — Extract Articles from New Newspapers

Once training is done:

**Process all images:**
```cmd
python step5_inference.py
```

**Process a single image:**
```cmd
python step5_inference.py --image output_images/lankadeepa_page_001.png
```

Output: `output/extracted_articles.json`

---

### STEP 8 — Evaluate Model Accuracy

```cmd
python step6_evaluate.py
```

Target F1 scores:
- headline:      > 0.90
- article_body:  > 0.88
- advertisement: > 0.85

If any class is below 0.80, annotate 50 more pages for that class and retrain.

---

### OPTIONAL — Visualize Predictions

Draw colored bounding boxes on a page to see what the model detects:

```cmd
python utils_visualize.py --image output_images/lankadeepa_page_001.png
```

Output saved to `output/lankadeepa_page_001_visualized.png`

**Color key:**
- 🟣 Purple  = headline
- 🟢 Teal    = article body
- 🟠 Coral   = advertisement
- 🩷 Pink    = image caption
- ⚫ Gray    = other

---

## Output Format

`output/extracted_articles.json`:

```json
{
  "lankadeepa_page_001.png": [
    {
      "headline": "ශ්‍රී ලංකාවේ ආර්ථිකය ශක්තිමත් වෙයි",
      "body": "ශ්‍රී ලංකා මහ බැංකුව අද ප්‍රකාශ කළේ..."
    },
    {
      "headline": "ක්‍රිකට් කණ්ඩායම ජය ගනී",
      "body": "ශ්‍රී ලංකා ක්‍රිකට් කණ්ඩායම ඊයේ..."
    }
  ]
}
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `conda not recognized` | Use Anaconda Prompt or CMD, not PowerShell |
| `tesseract not found` | Add `C:\Program Files\Tesseract-OCR` to PATH |
| `sin` not in tesseract langs | Reinstall Tesseract with Sinhala language pack checked |
| OCR produces garbage | Increase image DPI to 400 in step1 |
| CUDA out of memory | Keep `no_cuda=True` in step4, use CPU |
| Model not found in inference | Run step4_train.py first; training must complete |
| Low F1 score | Annotate more pages; focus on low-scoring class |

---

## Quick Reference — All Commands

```cmd
conda activate layoutlm
python step1_pdf_to_images.py
python step2_build_dataset.py
python step3_preprocess.py
python step4_train.py
python step5_inference.py
python step6_evaluate.py
```
