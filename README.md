[README (1).md](https://github.com/user-attachments/files/26413569/README.1.md)
# 🪪 Aadhaar OCR v5

**Maximum Field Extraction · QR Fraud Detection · Face AI · Groq LLM Correction**

A production-ready, fully offline-capable Aadhaar card intelligence pipeline. Extracts every field printed on an Aadhaar card, verifies authenticity via QR cross-check, and optionally matches the card photo against a live selfie — all in a single Python script with no paid APIs required.

---

## Table of Contents

- [What This Does](#what-this-does)
- [Every Field Extracted](#every-field-extracted)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [How It Works — Full Pipeline](#how-it-works--full-pipeline)
- [Library Reference](#library-reference)
- [QR Fraud Detection](#qr-fraud-detection)
- [Face AI Module](#face-ai-module)
- [LLM Error Correction via Groq](#llm-error-correction-via-groq)
- [Trust Score System](#trust-score-system)
- [Known Limitations](#known-limitations)
- [Applications You Can Build](#applications-you-can-build)

---

## What This Does

Real Aadhaar card photos from phones are blurry, compressed, tilted, and poorly lit. Standard OCR tools like Tesseract alone fail on them constantly. This pipeline solves that by stacking multiple engines, correcting their errors against each other, and adding a fraud detection layer that requires zero training data.

**In one run it:**

1. Preprocesses the image through 12 adaptive stages
2. Runs three OCR engines simultaneously and votes on the best output
3. Extracts 20+ individual fields including structured address components
4. Corrects character-level OCR errors using an LLM (Groq, free)
5. Decodes the embedded QR code through 15 preprocessing variants
6. Cross-checks QR data against OCR data to detect tampering
7. Extracts the face photo from the card, scores its quality, optionally matches it against a selfie
8. Outputs a Trust Score (0–100) and a clear verdict

---

## Every Field Extracted

### Identity
| Field | Description | Example |
|---|---|---|
| `name` | Full English name | `GHANSHYAM JETHARAM KUMAVAT` |
| `name_hindi` | Devanagari name below English | `घनश्याम जेठाराम कुमावत` |
| `dob` | Date of birth | `10/11/2005` |
| `year_of_birth` | Year only (if full DOB not present) | `1985` |
| `gender` | Normalized | `Male` / `Female` / `Transgender` |
| `father_husband_name` | C/O, S/O, W/O relationship name | `JETHARAM KUMAVAT` |

### Document Numbers
| Field | Format | Notes |
|---|---|---|
| `aadhaar_number` | `XXXX XXXX XXXX` | First digit always 2–9 per UIDAI spec |
| `vid` | `XXXX XXXX XXXX XXXX` | 16-digit Virtual ID (newer cards) |
| `enrollment_number` | `1234/12345/12345` | EID on old Aadhaar letters |

### Contact
| Field | Notes |
|---|---|
| `mobile` | 10-digit Indian mobile number |
| `email` | If printed (rare) |

### Address — Fully Structured
Every component is extracted individually rather than dumped as one blob:

| Field | Description |
|---|---|
| `address_house` | House / Door / Flat / Plot number |
| `address_street` | Street / Lane / Road |
| `address_landmark` | Near / Opp / Behind reference |
| `address_locality` | Village / Town / Nagar / Colony |
| `address_subdistrict` | Taluka / Tehsil / Mandal / Block |
| `address_district` | District name |
| `address_state` | Matched against all 36 Indian states and UTs |
| `address_pin` | 6-digit PIN code |
| `address_raw` | Full raw text fallback |

### Meta
| Field | Notes |
|---|---|
| `issue_date` | Card issue date |
| `card_type` | `Front Side` / `Back Side` / `e-Aadhaar / Combined` |
| `issued_by` | Always: Unique Identification Authority of India |

---

## Quick Start

```bash
# 1. Clone / download the project
git clone https://github.com/yourrepo/aadhaar-ocr

# 2. Install dependencies
pip install -r requirements.txt
sudo apt install tesseract-ocr tesseract-ocr-hin libzbar0   # Ubuntu/Debian

# 3. Set your free Groq API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env
# Get a free key at: https://console.groq.com

# 4. Run — file picker opens automatically
python aadhaar_ocr_v5.py

# 5. Run with selfie face matching
python aadhaar_ocr_v5.py --selfie path/to/selfie.jpg
```

When you run the script, **two file picker dialogs** open in sequence:
- **Step A** — Select the **front side** of the Aadhaar card (name, DOB, photo)
- **Step B** — Select the **back side** (address, QR code) — press Cancel to skip

---

## Installation

### System packages (required)

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr tesseract-ocr-hin libzbar0

# macOS
brew install tesseract tesseract-lang zbar

# Windows
# Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# zbar: bundled with pyzbar wheel on Windows
```

### Python packages

```bash
pip install -r requirements.txt
```

`requirements.txt` contents:

```
# Core — required
opencv-python
numpy
Pillow
pytesseract

# LLM correction via Groq (free)
openai
python-dotenv

# QR detection (strongly recommended)
pyzbar

# Face AI — deep learning face match (recommended)
deepface
tensorflow

# OCR Engine 2 — PaddleOCR
paddlepaddle
paddleocr

# OCR Engine 3 — TrOCR + local LLM fallback
transformers
torch
sentencepiece

# Super resolution (optional — for very blurry cards)
# realesrgan
# basicsr
```

### .env file

Create a `.env` file in the same folder as `aadhaar_ocr_v5.py`:

```
GROQ_API_KEY=gsk_your_key_here
```

> ⚠️ Add `.env` to your `.gitignore` — never commit API keys to Git.

---

## Project Structure

```
aadhaar_ocr_v5.py        Main script — 85 functions, ~4500 lines
requirements.txt         Python dependencies
.env                     Your API keys — never commit this
.gitignore               Should include: .env
README.md                This file
aadhaar_face_crop.jpg    Auto-saved face crop after each run
```

---

## How It Works — Full Pipeline

The pipeline runs front and back side images in parallel, then merges their OCR output before extraction.

```
FRONT IMAGE                          BACK IMAGE (optional)
     │                                     │
     ▼                                     ▼
Steps 1–12: Preprocessing           Steps 1–12: Preprocessing
(includes face photo masking)       (no face mask — back has no photo)
     │                                     │
     ▼                                     ▼
Step 13: Triple-Engine OCR          Step 13: Triple-Engine OCR
  ├─ PaddleOCR (primary)              ├─ PaddleOCR
  ├─ Tesseract 5-pass                 ├─ Tesseract 5-pass
  └─ TrOCR (blurry rescue)            └─ TrOCR
     │                                     │
     └──────────── MERGE ─────────────────┘
                    │
                    ▼
         Step 14:  Extract all 20+ fields
         Step 14B: Cross-pass error correction
         Step 14C: Groq LLM correction
         Step 15:  Field format verification
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
     Step 18:             Step 19:
     QR Verify            Face AI
     (back side first,    (front only)
     then front)          19A: Extract face
     15 variants          19B: Quality score
     Trust Score          19C: Selfie match
     Fraud signals        19D: Liveness hint
          └─────────┬──────────┘
                    ▼
         Step 17: Final Summary
```

### Preprocessing Steps (1–12)

| Step | Operation | Why It Matters |
|---|---|---|
| 1 | Load + validate | Check file exists, report dimensions |
| 2 | Resize to 1600px wide | Raises effective DPI to ~300 for Tesseract |
| 2B | Real-ESRGAN super-resolution | Only when Laplacian blur score < 30; optional |
| 3 | Auto-orient | Rotates portrait cards to landscape |
| 4 | Mask face photo region | Left 28% of card whited out — facial features read as garbled text otherwise |
| 5 | Remove saffron/green strips | Aadhaar color bands become unpredictable noise after binarization |
| 6 | Grayscale | Required for denoising and binarization |
| 7 | Non-local means denoise | Removes JPEG artifacts while preserving text stroke edges better than Gaussian |
| 8 | CLAHE contrast boost | Per-8×8-tile histogram equalization — fixes shadows and overexposure |
| 9 | Adaptive sharpening | 4 strategies by blur score: light USM / medium USM / Laplacian + high-pass / Wiener deconv |
| 10 | Binarize × 3 | Otsu + Adaptive Gaussian + Blended — 3 variants fed to different Tesseract passes |
| 11 | Deskew | minAreaRect angle detection, corrects tilt < 0.3° ignored |
| 12 | Morphological cleanup | CLOSE → OPEN → CLOSE with tiny kernels — closes char stroke gaps, removes specks |

### Triple OCR Engine (Step 13)

**Engine 1: PaddleOCR** (primary)
Uses a three-stage deep learning pipeline: DB (Differentiable Binarization) text detection → direction classifier → CRNN recognition. Trained on millions of real document images. Handles multilingual text, tilted/curved layout, and low-resolution inputs far better than Tesseract. PaddleOCR v3+ API changes (`use_gpu`, `use_angle_cls`, `show_log` removed) are handled with a 3-attempt graceful init.

**Engine 2: Tesseract 5** (supplementary, 5 passes)
Runs configurations `psm 6`, `psm 4`, `psm 11` with `eng` and `hin+eng` language models. Each pass produces different errors. All 5 outputs are combined and fed into the cross-pass voter.

**Engine 3: TrOCR** (rescue, conditional)
Microsoft's Vision Transformer OCR. Activated only when total extracted characters < 80 (confidence gate). Uses attention over the full image context to read text that CRNN-based engines fail on in severely blurry images. Downloads ~1.3GB `trocr-large-printed` model on first use.

### Name Extraction — 3-Strategy Cascade

Because name extraction was a key source of bugs, it uses three strategies in order:

**Strategy 1 — Positional anchor** (most reliable)
Every genuine Aadhaar front side has "Government of India" at the top. The name always appears in the next 12 lines, before the DOB line. The code searches all occurrences of this header in the merged multi-pass text (each pass contributes a GOI line).

**Strategy 2 — Label anchor**
Searches for explicit `Name:`, `Full Name:`, or `नाम:` prefixes.

**Strategy 3 — Scoring fallback**
Every line is scored: ALL_CAPS (+10), 2–4 words (+8), all words ≥ 3 chars (+3), 3-part name (+5), between GOI and DOB positionally (+4 each). Minimum 5 characters and 2 words required — prevents single noise words like `LEE` or `ACT` from winning.

### Error Correction (Step 14B)

**DOB voting**: Each OCR pass produces `(day, month, year)` tuples. All tuples from all passes are collected and the most common value for each component wins independently. Handles fused readings like `"12006"` (month=1, year=2006) and `"172008"` (noise+year).

**Name voting**: Same-structure voting — only candidates with the same word count and same first letter per word are included. Prevents garbage lines from other image regions corrupting the name.

---

## Library Reference

| Library | Role | Why This One |
|---|---|---|
| `opencv-python` | All image processing, Haar face detection, QR fallback | Only serious choice — 20yr history, C++ backend, covers everything |
| `numpy` | Array math, FFT, histograms | Universal substrate — all other libs use it |
| `pytesseract` | Tesseract wrapper | Thin wrapper for Tesseract 5 LSTM engine |
| `Pillow` | PIL image I/O for Tesseract/TrOCR passes | Required by both pytesseract and transformers |
| `paddleocr` + `paddlepaddle` | Primary DL OCR engine | Better than Tesseract on real-world noisy docs; better than EasyOCR on document layout |
| `transformers` + `torch` | TrOCR rescue engine + flan-t5 local LLM | TrOCR attention mechanism handles blur that CRNN cannot |
| `deepface` + `tensorflow` | Face match (VGG-Face, 98.95% LFW accuracy) | Unified API over 6 face recognition models; auto face alignment |
| `pyzbar` | Primary QR decoder (more robust than OpenCV's) | ZBar C backend handles blurry/partial QR better than OpenCV |
| `openai` | Groq API client (Groq uses OpenAI-compatible format) | One package works for both OpenAI and Groq |
| `python-dotenv` | Load `GROQ_API_KEY` from `.env` file | Prevents hardcoded secrets in code |

**Why not EasyOCR instead of PaddleOCR?**
Both use CRNN architecture. PaddleOCR is faster at inference, has better Hindi support, and is designed for structured document layout. EasyOCR is better for scene text (signboards, photos). For ID card documents, PaddleOCR wins.

**Why not OpenAI/Anthropic instead of Groq?**
Groq's LPU hardware runs LLaMA 3.3 70B at 500+ tokens/sec — faster than GPT-4o and ~10× faster than local inference. Free tier (14,400 req/day) is sufficient for any development workload. No credit card required.

**Why not ArcFace instead of VGG-Face?**
ArcFace achieves 99.83% on LFW vs VGG-Face's 98.95%. However, DeepFace supports ArcFace as a one-parameter switch (`model_name='ArcFace'`) — change it in `step19c_face_match` if you need the extra 0.88%.

---

## QR Fraud Detection

Every genuine Aadhaar card has a QR code written at enrollment time by UIDAI. A tampered document either:
- Has no valid QR → detected immediately (trust score 35)
- Has a QR that doesn't match the edited text → field mismatch detected

This gives fraud detection with **zero training data, no UIDAI API, fully offline**.

### 15-Variant Preprocessing Pipeline

Before attempting QR detection, the image is processed through 15 variants, stopping at the first successful decode:

| # | Variant | Best For |
|---|---|---|
| 1 | Raw color | Already-clear images |
| 2 | CLAHE | Low contrast |
| 3 | Unsharp mask | Mild blur |
| 4 | CLAHE + USM | Combined |
| 5 | Wiener deconv (5px) | Camera shake |
| 6 | Wiener deconv (9px) | Severe blur |
| 7 | Otsu binary | Good contrast |
| 8 | Adaptive threshold | Shadows/uneven light |
| 9 | Morph open | Noise/artifacts |
| 10–11 | 1.5× upscale + USM/Otsu | Small QR in frame |
| 12–13 | 2× upscale + USM/Otsu | Low-res photo |
| 14–15 | 3× upscale + USM/Otsu | Very low-res / far shot |

The result shows which variant succeeded: `QR Source: pyzbar [BACK / up2x+otsu]`

### Three QR Payload Formats

**Format A — XML** (cards printed before ~2018):
```xml
<PrintLetterBarcodeData uid="2102 6671 5472"
  name="GHANSHYAM JETHARAM KUMAVAT" dob="10/11/2005" gender="M"
  co="JETHARAM KUMAVAT" house="2" street="POONAM NIVAS ROOM NO 2"
  dist="Thane" state="Maharashtra" pc="421306"/>
```

**Format B — Pipe-delimited** (cards from ~2018 onwards):
```
2|5472|GHANSHYAM JETHARAM KUMAVAT|10/11/2005|M|JETHARAM|2|...|Thane|Maharashtra|421306|7890
```

**Format C — JSON** (rare, state-specific variants):
```json
{"name": "GHANSHYAM JETHARAM KUMAVAT", "dob": "10/11/2005", "uid": "2102 6671 5472"}
```

### Trust Score

| Signal | Points |
|---|---|
| QR code found | +20 |
| QR successfully decoded | +10 |
| Aadhaar number matches | +35 |
| Name matches (bigram Jaccard ≥ 75%) | +20 |
| DOB matches | +10 |
| Gender matches | +5 |
| **Maximum** | **100** |

| Score | Verdict |
|---|---|
| ≥ 80 | ✅ LIKELY GENUINE |
| 50–79 | ⚠️ REVIEW REQUIRED |
| < 50 | ❌ FRAUD SUSPECTED |
| QR not found | ⚠️ REVIEW REQUIRED (QR not found) — photograph back side clearly |

---

## Face AI Module

Runs on the front side image only (which contains the photo). Four stages:

### 19A — Face Extraction
Crops left 28% × vertical 12–88% of card (UIDAI fixed layout). Runs Haar Cascade face detection at 4 scale factors within that region. Adds 20% padding around detected face. Falls back to full region crop if detection fails. Saves the face crop as `aadhaar_face_crop.jpg`.

### 19B — Face Quality Assessment

| Metric | Measurement | Max Score |
|---|---|---|
| Sharpness | Laplacian variance | 35 |
| Brightness | Mean pixel (target 60–200) | 25 |
| Size | Pixel area (target > 5000px²) | 25 |
| Contrast | Pixel std dev | 15 |

Score < 45 = **POOR** — match result should not be trusted regardless of verdict.

### 19C — Selfie Matching

**Primary: DeepFace + VGG-Face**
```
card_face → VGG-Face embedding (2622-dim)
selfie    → VGG-Face embedding (2622-dim)
cosine_distance = 1 − (a·b / |a||b|)
MATCH if distance < 0.40
```
VGG-Face is trained on 2.6M celebrity images, achieves 98.95% on the LFW benchmark. To use ArcFace (99.83% accuracy), change `model_name='ArcFace'` in `step19c_face_match`.

**Fallback: OpenCV LBPH**
When DeepFace not installed. Divides face into 4×4 grid, computes LBP histogram per cell, Chi-squared comparison. Less accurate but zero extra dependencies.

LBPH thresholds:
- Score ≥ 70 → **MATCH** (strong)
- Score ≥ 55 → **MATCH** (weak — verify manually)
- Score ≥ 35 → **INCONCLUSIVE** (install deepface for better accuracy)
- Score < 35 → **NO MATCH**

### 19D — Liveness Hint

Detects "photo of a photo" attacks via 2D FFT frequency analysis. Printed photos have regular halftone dot patterns → sharp periodic peaks in frequency domain. Phone screens have pixel grid → strong energy at specific frequencies. Real skin has natural distributed texture → no dominant peaks.

Liveness score (0–100) is penalized for high peak ratios. This is a heuristic — not a certified Presentation Attack Detection system. For production PAD, integrate `Silent-Face-Anti-Spoofing`.

---

## LLM Error Correction via Groq

### Why an LLM for OCR correction

Regex rules only fix errors you've already catalogued. An LLM trained on billions of documents knows:
- `GHANSHVAN` is almost certainly `GHANSHYAM` (Hindi name suffix pattern)
- `KUMAWAT` is likely `KUMAVAT` (common OCR suffix confusion)
- If `JETHARAM` appears 3× in the raw text and `JETHARAN` once, the former is correct

### Setup (2 minutes, free)

```bash
# 1. Get free API key
# Visit: https://console.groq.com → API Keys → Create

# 2. Add to .env file
GROQ_API_KEY=gsk_your_key_here

# 3. Install openai package (Groq uses same API format)
pip install openai
```

Free tier limits: 14,400 requests/day, 6,000 tokens/minute — more than enough for development.

### Safety Guardrails

The LLM cannot hallucinate a completely different name because:

1. The prompt includes the raw OCR text — the LLM sees `GHANSHYAM` appearing multiple times and uses that as evidence
2. Post-processing rejects any suggested name where more than 1 word starts with a different first letter than the current name (e.g., if OCR extracted G-J-K names, only G-J-K suggestions are accepted)
3. `null` values are ignored — the LLM can only modify, never invent from nothing

### Fallback chain

```
GROQ_API_KEY set?
  YES → Groq API (llama-3.3-70b-versatile)
  NO  → local google/flan-t5-small (~80MB, downloads once)
  Neither → fields returned unchanged (never crashes)
```

---

## Trust Score System

**OCR Verification** (format checks):
- ✓ Aadhaar: 12 digits, first digit 2–9
- ✓ Name: only English letters/spaces/hyphens/dots
- ✓ DOB: valid date, year 1900–2025
- ✓ Gender: exactly Male / Female / Transgender

**QR Trust Score** (fraud detection, 0–100):
See [Trust Score](#trust-score) table above.

**Combined reading:**
```
QR ≥ 80 + all OCR valid   →  HIGH CONFIDENCE GENUINE
QR ≥ 80 + OCR has issues  →  GENUINE — review OCR fields manually
QR 50–79                  →  REVIEW REQUIRED
QR < 50                   →  FRAUD SUSPECTED — escalate
QR not found              →  INCONCLUSIVE — photograph back side clearly
```

---

## Known Limitations

| Issue | Impact | Workaround |
|---|---|---|
| PaddleOCR v3.x API changes | `use_gpu`, `show_log`, `use_angle_cls` removed | 3-attempt graceful init cascade (fixed in v5) |
| Very blurry card photos | Lower OCR accuracy overall | TrOCR rescue engine + Wiener deconv help significantly |
| Hindi OCR accuracy | ~70% vs ~95% for English Tesseract | Used as supplementary field only |
| WhatsApp-compressed back side | JPEG artifacts break QR modules | 15-variant QR preprocessing recovers most cases |
| Very small face photo | Haar Cascade may miss it | Falls back to full left-region crop |
| Liveness detection | Heuristic FFT analysis, not certified PAD | Add Silent-Face-Anti-Spoofing for production |
| Address parsing | Depends on keyword prefix presence | Raw address always available; structured fields are best-effort |
| Legal compliance | Not a UIDAI-licensed eKYC system | Research/demo use only without proper licensing |

---

## Applications You Can Build

| Application | Key Pipeline Features Used |
|---|---|
| **eKYC API for fintechs / NBFCs** | All 20+ fields + QR Trust Score as JSON from a FastAPI endpoint |
| **HR onboarding automation** | Auto-fill HRMS fields from card photo; flag duplicate enrollments |
| **Rental tenant verification** | Address PIN + district extraction; QR authenticity check |
| **Hotel / hostel digital check-in kiosk** | Front+back processing + real-time selfie face match via webcam |
| **Telemedicine patient identity** | Face match card vs live camera before consultation |
| **Anti-duplicate subsidy detection** | Store face embeddings in pgvector/Qdrant; ANN search for duplicates |
| **Multi-document KYC** | Extend same pipeline to PAN card, Voter ID, Driving Licence |
| **OCR accuracy research** | 3-engine comparison benchmark on Indian ID documents |
| **Synthetic fraud dataset generation** | Generate fake cards with altered QR; train downstream classifiers |

---

## Version History

| Version | Key Additions |
|---|---|
| v1 | Basic Tesseract, name + Aadhaar number + DOB |
| v2 | PaddleOCR primary engine |
| v3 | Multi-pass voting, adaptive sharpening (4 strategies), Wiener deconv |
| v4 | QR cross-verification + Trust Score, improved name extraction with positional anchoring |
| v5 | All 20+ fields, Hindi name, VID, enrollment number, structured address, Face AI (19A–19D), front+back dual input, Groq LLM correction, 15-variant QR preprocessing, correct_name voting fix, PaddleOCR v3 compatibility |

---

## Legal & Ethics

This project is for **research and educational purposes only**.

- Do not store Aadhaar numbers without explicit user consent
- Comply with India's Digital Personal Data Protection Act 2023 (DPDP Act)
- Production eKYC using Aadhaar requires UIDAI AUA/KUA licensing
- All extracted data is PII under Indian law — handle with appropriate safeguards
- Do not use for mass surveillance or unauthorized identity verification

---

*Built with OpenCV · Tesseract · PaddleOCR · TrOCR · DeepFace · pyzbar · Groq LLaMA 3.3*
