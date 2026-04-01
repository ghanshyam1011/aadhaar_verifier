# main.py
# Entry point — ties all modules together.
# Run: python main.py
#      python main.py --selfie selfie.jpg
# ─────────────────────────────────────────────────────────────
# No code changed — same functions, split into own file.

import os
import sys

from numpy import rint
from codecarbon import EmissionsTracker

from utils import section, ok, info, warn, err

from preprocessing import (
    step1_load,
    step2_resize,
    step2b_super_resolution,
    step3_orient,
    step4_mask_photo,
    step5_remove_color_noise,
    step6_grayscale,
    step7_denoise,
    step8_clahe,
    step9_adaptive_sharpen,
    step10_binarize,
    step11_deskew,
    step12_morph,
)

from ocr_engines import (
    step13_tesseract,
    run_trocr,
)

from field_extraction import step14_extract

from ocr_correction import (
    step14b_correct,
    step14c_llm_correct,
)

from face_ai import step19_face_pipeline

from qr_verification import step18_qr_verify

from verification_summary import (
    step15_verify,
    step17_summary,
)



# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def open_file_dialog():
    """
    Opens a native OS file picker dialog so the user can
    click and select their Aadhaar card image visually.

    Works in 3 environments:
      1. Desktop (Windows/Mac/Linux) : tkinter file dialog window
      2. Google Colab                : google.colab files.upload()
      3. Jupyter Notebook            : ipywidgets FileUpload widget
      4. Fallback                    : typed path input (terminal)
    """
    # ── Try Google Colab ──────────────────────────────────
    try:
        import google.colab
        from google.colab import files as colab_files
        print("\n  [..] Google Colab detected.")
        print("  [..] A file upload button will appear below.")
        print("  [..] Click it and select your Aadhaar card image.\n")
        uploaded = colab_files.upload()
        if not uploaded:
            err("No file was uploaded.")
            sys.exit(1)
        filename = list(uploaded.keys())[0]
        ok(f"File uploaded via Colab: {filename}")
        return filename
    except (ImportError, ModuleNotFoundError):
        pass

    # ── Try tkinter desktop file dialog ──────────────────
    try:
        import tkinter as tk
        from tkinter import filedialog

        print("\n  [..] Opening file picker window...")
        print("  [..] (A dialog box will appear — select your image)")

        root = tk.Tk()
        root.withdraw()           # Hide the empty Tk root window
        root.attributes('-topmost', True)   # Dialog appears on top

        file_path = filedialog.askopenfilename(
            title="Select your Aadhaar Card Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("JPEG",        "*.jpg *.jpeg"),
                ("PNG",         "*.png"),
                ("All files",   "*.*"),
            ]
        )
        root.destroy()

        if not file_path:
            err("No file was selected. Please run again and select an image.")
            sys.exit(1)

        ok(f"File selected: {file_path}")
        return file_path

    except Exception as e:
        warn(f"File dialog failed ({e})")
        warn("Falling back to manual path input...")

    # ── Fallback: type the path manually ─────────────────
    print()
    print("  Please type or paste the path to your Aadhaar card image.")
    print("  Supported: JPG, JPEG, PNG, WEBP, BMP")
    print()
    while True:
        try:
            path = input("  >>> Image path: ").strip().strip('"\'')
        except EOFError:
            err("No input received.")
            sys.exit(1)
        if not path:
            print("  [!!]  Path cannot be empty. Try again.\n")
            continue
        if not os.path.exists(path):
            print(f"  [!!]  File not found: '{path}'. Try again.\n")
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            print(f"  [!!]  Unsupported format '{ext}'. Use JPG/PNG/WEBP.\n")
            continue
        ok(f"Image accepted: {path}")
        return path

def run_pipeline(front_path, back_path=None, selfie_path=None):

    tracker = EmissionsTracker()
    tracker.start()

    DOC_TYPE = "aadhaar"

    front_orig  = step1_load(front_path)
    front_res   = step2_resize(front_orig)
    front_sr    = step2b_super_resolution(front_res)
    front_ori   = step3_orient(front_sr)
    front_mask  = step4_mask_photo(front_ori)
    front_nc    = step5_remove_color_noise(front_mask)
    front_gray  = step6_grayscale(front_nc)
    front_den   = step7_denoise(front_gray)
    front_enh   = step8_clahe(front_den)
    front_sh    = step9_adaptive_sharpen(front_enh)

    f_otsu, f_adap, f_blend = step10_binarize(front_sh)
    front_desk  = step11_deskew(f_blend)
    front_clean = step12_morph(front_desk)

    front_text, front_passes = step13_tesseract(front_clean, front_ori, front_path)

    combined_text = front_text
    all_pass_texts = front_passes

    fields = step14_extract(combined_text)
    fields = step14b_correct(fields, all_pass_texts)

    verification, all_valid = step15_verify(fields)

    face_result = step19_face_pipeline(front_orig, front_path,
                                       selfie_path=selfie_path)

    qr_result = step18_qr_verify(front_path, fields)

    result = {
        "fields": fields,
        "verdict": {
            "label": "VALID" if all_valid else "REVIEW REQUIRED",
            "detail": "Verification completed",
            "score": 80 if all_valid else 40,
            "color": "green" if all_valid else "yellow",
            "passed": [],
            "issues": []
        },
        "qr": qr_result,
        "face": face_result
    }
    emissions = tracker.stop()

    print("\n🌱 Carbon Emission Report")
    print(f"Estimated CO₂ emissions: {emissions:.6f} kg")

    return result

def main():
    tracker = EmissionsTracker()
    tracker.start()
    print("\n" + "=" * 62)
    print("   AADHAAR CARD OCR v5 — FULL EXTRACTION + FACE AI")
    print("=" * 62)

    # ── Parse optional --selfie argument ─────────────────────
    selfie_path = None
    args = sys.argv[1:]
    if '--selfie' in args:
        idx = args.index('--selfie')
        if idx + 1 < len(args):
            selfie_path = args[idx + 1]
            ok(f"Selfie for face matching: {selfie_path}")
        else:
            warn("--selfie flag provided but no path given")
    if selfie_path and not os.path.exists(selfie_path):
        warn(f"Selfie file not found: {selfie_path}")
        selfie_path = None

    DOC_TYPE = "aadhaar"

    # ── Ask for FRONT side ────────────────────────────────────
    print()
    print("  ┌─────────────────────────────────────────────┐")
    print("  │  STEP A — Select FRONT of Aadhaar card      │")
    print("  │  (has: Name, DOB, Gender, Aadhaar Number)   │")
    print("  └─────────────────────────────────────────────┘")
    front_path = open_file_dialog()
    print(f"\n  Front : {front_path}")

    # ── Ask for BACK side ─────────────────────────────────────
    print()
    print("  ┌─────────────────────────────────────────────┐")
    print("  │  STEP B — Select BACK of Aadhaar card       │")
    print("  │  (has: Address, PIN, QR Code, Barcode)      │")
    print("  │  Press Cancel / Enter blank to skip         │")
    print("  └─────────────────────────────────────────────┘")
    try:
        back_path = open_file_dialog()
        if not back_path or not os.path.exists(back_path):
            back_path = None
    except SystemExit:
        back_path = None
    if back_path:
        print(f"\n  Back  : {back_path}")
    else:
        info("Back side skipped — address fields will be limited")

    # ──────────────────────────────────────────────────────────
    #  Process FRONT side
    # ──────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  PROCESSING FRONT SIDE")
    print("=" * 62)

    front_orig  = step1_load(front_path)
    front_res   = step2_resize(front_orig)
    front_sr    = step2b_super_resolution(front_res)
    front_ori   = step3_orient(front_sr)
    front_mask  = step4_mask_photo(front_ori)
    front_nc    = step5_remove_color_noise(front_mask)
    front_gray  = step6_grayscale(front_nc)
    front_den   = step7_denoise(front_gray)
    front_enh   = step8_clahe(front_den)
    front_sh    = step9_adaptive_sharpen(front_enh)
    f_otsu, f_adap, f_blend = step10_binarize(front_sh)
    front_desk  = step11_deskew(f_blend)
    front_clean = step12_morph(front_desk)

    front_text, front_passes = step13_tesseract(front_clean, front_ori, front_path)

    # ──────────────────────────────────────────────────────────
    #  Process BACK side (if provided)
    # ──────────────────────────────────────────────────────────
    back_text   = ""
    back_passes = []
    back_orig   = None

    if back_path:
        print()
        print("=" * 62)
        print("  PROCESSING BACK SIDE")
        print("=" * 62)

        back_orig   = step1_load(back_path)
        back_res    = step2_resize(back_orig)
        back_sr     = step2b_super_resolution(back_res)
        back_ori    = step3_orient(back_sr)
        # Don't mask photo on back — no face photo on back side
        back_nc     = step5_remove_color_noise(back_ori)
        back_gray   = step6_grayscale(back_nc)
        back_den    = step7_denoise(back_gray)
        back_enh    = step8_clahe(back_den)
        back_sh     = step9_adaptive_sharpen(back_enh)
        b_otsu, b_adap, b_blend = step10_binarize(back_sh)
        back_desk   = step11_deskew(b_blend)
        back_clean  = step12_morph(back_desk)

        back_text, back_passes = step13_tesseract(back_clean, back_ori, back_path)

    # ──────────────────────────────────────────────────────────
    #  Merge front + back OCR results
    # ──────────────────────────────────────────────────────────
    combined_text  = front_text + ("\n" + back_text if back_text else "")
    all_pass_texts = front_passes + back_passes

    ok(f"Total OCR text: {len(combined_text)} chars "
       f"(front={len(front_text)}, back={len(back_text)})")

    # ── Extract all fields from merged text ───────────────────
    fields = step14_extract(combined_text)

    # ── Correct OCR errors ────────────────────────────────────
    fields = step14b_correct(fields, all_pass_texts)

    # ── TrOCR pass (runs before LLM so LLM gets max context) ─
    info("Running TrOCR for additional OCR diversity...")
    trocr_text, trocr_conf, trocr_ok = run_trocr(front_path, confidence_threshold=0.5)
    if trocr_ok and trocr_text.strip():
        combined_text  = combined_text + "\n" + trocr_text
        all_pass_texts = all_pass_texts + [trocr_text]
        ok(f"TrOCR added {len(trocr_text)} chars (conf={trocr_conf:.3f})")
    elif not trocr_ok:
        info("TrOCR not available — install: pip install transformers torch")

    # ── Groq LLM correction (gets full combined_text as context) ─
    fields = step14c_llm_correct(fields, raw_ocr_text=combined_text)

    # ── Verify OCR fields ─────────────────────────────────────
    verification, all_valid = step15_verify(fields)

    # ── Face AI — run on front side (has the photo) ───────────
    face_result = step19_face_pipeline(front_orig, front_path,
                                       selfie_path=selfie_path)

    # ── QR Cross-Verification — prefer back side QR ──────────
    # Back side has the large QR code; front has a smaller one
    qr_source = back_path if back_path else front_path
    qr_result = step18_qr_verify(qr_source, fields)

    # ── Final Summary ─────────────────────────────────────────
    step17_summary(fields, verification, all_valid,
                   qr_result=qr_result, face_result=face_result)
    emissions = tracker.stop()

    print("\n🌱 Carbon Emission Report")
    print(f"Estimated CO₂ emissions: {emissions:.6f} kg")

if __name__ == "__main__":
    main()