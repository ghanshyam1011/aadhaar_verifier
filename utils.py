"""
============================================================
  Aadhaar Card OCR v5 — Maximum Field Extraction + Face AI
  LLM: Groq API (llama-3.3-70b) — free tier, no Anthropic needed

  EVERY FIELD EXTRACTED FROM AADHAAR:
    ✦ Name (English)           ✦ Name (Hindi/Devanagari)
    ✦ Date of Birth            ✦ Year of Birth (partial DOB)
    ✦ Gender                   ✦ Aadhaar Number (12-digit)
    ✦ Virtual ID (VID)         ✦ Enrollment Number
    ✦ Father / Husband name    ✦ Mobile Number
    ✦ Email Address            ✦ Full Address (structured)
      → House/Door No          → Street / Lane
      → Landmark               → Village / Town / City
      → Sub-District           → District
      → State                  → PIN Code
    ✦ Issue Date               ✦ Card Type (front/back)
    ✦ QR Code data (all fields from embedded QR)

  FACE AI MODULE (Step 19):
    ✦ Face Extraction     — crops the photo from card
    ✦ Face Quality Check  — blur/brightness/size scoring
    ✦ Face Detection      — verify a face actually exists
    ✦ Selfie Matching     — compare card photo vs live selfie
      → Uses DeepFace (deep learning face recognition)
      → Falls back to OpenCV LBPH if DeepFace not installed
      → Returns similarity score + MATCH / NO MATCH verdict
    ✦ Liveness Hints      — basic texture analysis to flag
      printed-photo-of-photo attacks

  QR VERIFICATION (Step 18 from v4):
    ✦ QR decode + cross-check vs OCR fields
    ✦ Trust Score 0–100 with fraud signals

INSTALL:
    pip install opencv-python pillow pytesseract numpy openai
    pip install deepface        (optional — deep learning face match)
    pip install pyzbar          (optional — better QR detection)
    Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-hin libzbar0

LLM SETUP (free):
    1. Get free Groq key: https://console.groq.com
    2. pip install python-dotenv
    3. Create .env file in same folder as this script:
         GROQ_API_KEY=gsk_your_key_here
    (Never hardcode your key — .env keeps it safe)

RUN:
    python aadhaar_ocr_v5.py
    python aadhaar_ocr_v5.py --selfie selfie.jpg
============================================================
"""

import cv2
import numpy as np
import pytesseract
import argparse
import re
import os
import sys

# ── Load .env file if present ─────────────────────────────────
# Create a .env file in the same folder as this script with:
#   GROQ_API_KEY=gsk_...
# pip install python-dotenv
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        print(f"  [OK]  .env loaded from: {_env_path}")
    else:
        load_dotenv()   # also tries current working directory
except ImportError:
    pass   # dotenv not installed — os.environ still works normally


# ── OCR Digit Fix Helper ─────────────────────────────

DIGIT_FIX = str.maketrans({
    'O': '0', 'o': '0', 'D': '0', 'Q': '0',
    'I': '1', 'l': '1', 'i': '1', '|': '1',
    'Z': '2', 'z': '2',
    'S': '5', 's': '5',
    'G': '6', 'b': '6',
    'T': '7',
    'B': '8',
    'g': '9', 'q': '9',
})

def fix_digit_string(s):
    """
    Fix OCR mistakes where letters look like digits.
    Example:
        O -> 0
        l -> 1
        S -> 5
        B -> 8
    """
    if not s:
        return s
    return s.translate(DIGIT_FIX)

# ─────────────────────────────────────────────────────────────
#  PRINT HELPERS
# ─────────────────────────────────────────────────────────────

def section(title):
    print("\n" + "=" * 62)
    print(f"  STEP: {title}")
    print("=" * 62)

def ok(msg):   print(f"  [OK]  {msg}")
def info(msg): print(f"  [..]  {msg}")
def warn(msg): print(f"  [!!]  {msg}")
def err(msg):  print(f"  [XX]  {msg}")

def extract_dob_tuples(text):
    """
    Same as extract_dob but returns ALL (day,month,year) tuples found.
    Used by the voter to get maximum candidates from each pass.
    """
    tuples = []

    def try_add(d_s, m_s, y_s):
        try:
            d = int(fix_digit_string(str(d_s).strip()))
            m = int(fix_digit_string(str(m_s).strip()))
            y = int(fix_digit_string(str(y_s).strip()))
            if 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2025:
                tuples.append((d, m, y))
        except (ValueError, TypeError):
            pass

    for mt in re.finditer(r'(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})', text):
        try_add(mt.group(1), mt.group(2), mt.group(3))

    m = re.search(
        r'(?:DOB|DOR|D0B|D08|Date\s*of\s*Birth)[:\s]*'
        r'(\d{1,2})[/.\- ]{0,2}(\d{1,2})[/.\- ]{0,2}(\d{4})',
        text, re.IGNORECASE)
    if m:
        try_add(m.group(1), m.group(2), m.group(3))

    m = re.search(
        r'(?:DOB|DOR|D0B|D08)[:\s]+(\d{1,2})\s+(\d)((?:19|20)\d{2})',
        text, re.IGNORECASE)
    if m:
        try_add(m.group(1), m.group(2), m.group(3))

    m = re.search(
        r'(?:DOB|DOR|D0B|D08)[:\s]*([\d/ .\-]{4,20})',
        text, re.IGNORECASE)
    if m:
        raw = m.group(1)
        clusters = [fix_digit_string(c) for c in re.findall(r'\d+', raw)]
        if len(clusters) >= 3:
            try_add(clusters[0], clusters[1], clusters[2])
        if len(clusters) >= 2 and len(clusters[1]) == 5:
            try_add(clusters[0], clusters[1][0], clusters[1][1:])
        if len(clusters) == 2 and len(clusters[1]) == 6:
            try_add(clusters[0], clusters[1][:2], clusters[1][2:])

    # Deduplicate
    seen = set()
    result = []
    for t in tuples:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result
