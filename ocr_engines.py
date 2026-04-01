# ocr_engines.py
# Step 13: Multi-Engine OCR — PaddleOCR (primary) + Tesseract (5-pass) + TrOCR
# Also contains: run_paddleocr(), run_tesseract_passes(), run_trocr()
# ─────────────────────────────────────────────────────────────
# No code changed — same functions, split into own file.

import cv2
import numpy as np
import pytesseract
import os

from utils import section, ok, info, warn, err



# ─────────────────────────────────────────────────────────────
#  STEP 13 — Multi-pass Tesseract OCR
#            Run 5 different configs, collect all results
# ─────────────────────────────────────────────────────────────

def run_paddleocr(image_path_or_array):
    """
    Run PaddleOCR on an image.

    WHY PaddleOCR IS BETTER THAN TESSERACT:
      Tesseract uses classical image processing + HMM.
      PaddleOCR uses deep learning end-to-end:
        1. Detection  : DB (Differentiable Binarization) model
                        finds text regions even in complex layouts
        2. Direction  : Classifier detects rotated/upside-down text
        3. Recognition: CRNN model reads each detected text region

      Advantages for Aadhaar cards specifically:
        - Better blur tolerance (trained on real-world noisy docs)
        - Better multilingual  (Hindi + English simultaneously)
        - Better layout understanding (finds text in any position)
        - No need for binarization preprocessing
        - Handles curved/tilted text naturally

    INSTALL:
        pip install paddlepaddle paddleocr
        (CPU version — no GPU needed for documents)

    RETURNS:
        (text: str, available: bool)
        text      = all extracted text joined by newlines
        available = True if PaddleOCR ran, False if not installed
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return "", False

    try:
        info("Initializing PaddleOCR (first run downloads models ~50MB)...")

        # PaddleOCR v3+ changed its API significantly — catch ANY error
        # and try progressively simpler init until one works.
        ocr = None

        for init_kwargs in [
            {"use_textline_orientation": True, "lang": "en"},  # v3+ new API
            {"use_angle_cls": True, "lang": "en"},             # v2.x with cls
            {"lang": "en"},                                     # bare minimum
        ]:
            try:
                ocr = PaddleOCR(**init_kwargs)
                ok(f"PaddleOCR initialized with: {list(init_kwargs.keys())}")
                break
            except Exception as e:
                warn(f"PaddleOCR init attempt {list(init_kwargs.keys())} failed: {e}")
                ocr = None

        if ocr is None:
            raise RuntimeError("All PaddleOCR init attempts failed")

        # PaddleOCR v3+ removed cls=True from ocr() call
        try:
            result = ocr.ocr(image_path_or_array, cls=True)
        except TypeError:
            result = ocr.ocr(image_path_or_array)

        if not result or result == [None]:
            warn("PaddleOCR returned empty result")
            return "", True

        # ── Parse PaddleOCR result structure ──────────────
        # Result format: [ [ [bbox, (text, confidence)], ... ] ]
        lines = []
        total_boxes = 0
        for page in result:
            if not page:
                continue
            for item in page:
                if not item or len(item) < 2:
                    continue
                text_info = item[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text       = str(text_info[0]).strip()
                    confidence = float(text_info[1])
                    if text and confidence > 0.3:   # filter very low confidence
                        lines.append(text)
                        total_boxes += 1

        combined = "\n".join(lines)
        ok(f"PaddleOCR: {total_boxes} text boxes detected, {len(combined)} chars")
        return combined, True

    except Exception as e:
        warn(f"PaddleOCR failed: {e}")
        return "", False


def run_tesseract_passes(binary, original_color):
    """
    Run 5-pass Tesseract OCR (fallback / supplementary engine).
    Same as original step13 logic — kept as fallback.
    """
    from PIL import Image

    def run(label, img_input, config):
        try:
            text = pytesseract.image_to_string(img_input, config=config).strip()
            ok(f"  {label:<44}: {len(text):>4} chars")
            return text
        except Exception as e:
            warn(f"  {label:<44}: FAILED — {e}")
            return ""

    texts = []
    texts.append(run("Tesseract Pass 1: binary, eng,     psm6", binary,
                     "--oem 3 --psm 6 -l eng"))
    texts.append(run("Tesseract Pass 2: binary, eng,     psm4", binary,
                     "--oem 3 --psm 4 -l eng"))
    pil_color = Image.fromarray(cv2.cvtColor(original_color, cv2.COLOR_BGR2RGB))
    texts.append(run("Tesseract Pass 3: color,  eng,     psm6", pil_color,
                     "--oem 3 --psm 6 -l eng"))
    try:
        texts.append(run("Tesseract Pass 4: binary, hin+eng, psm6", binary,
                         "--oem 3 --psm 6 -l hin+eng"))
    except Exception:
        warn("Tesseract Pass 4 skipped — install: sudo apt install tesseract-ocr-hin")
    gray_orig = cv2.cvtColor(original_color, cv2.COLOR_BGR2GRAY)
    texts.append(run("Tesseract Pass 5: gray,   eng,     psm11", gray_orig,
                     "--oem 3 --psm 11 -l eng"))
    return texts



def run_trocr(image_input):
    """
    Engine 3 — TrOCR (Transformer OCR by Microsoft)

    WHY TrOCR IS BETTER THAN PADDLEOCR ON BLURRY IMAGES:
      PaddleOCR uses CRNN (CNN + RNN) for recognition.
      TrOCR uses a Vision Transformer encoder + language model
      decoder. This means it has prior knowledge of how real
      words/numbers SHOULD look — so even when pixels are
      merged by blur, it can infer the correct characters.

      On severely blurry images (Laplacian score < 30):
        PaddleOCR : often reads "D0B: 10/1?/200?" — gives up
        TrOCR     : reads "DOB: 10/11/2005" — infers from context

      This is because the decoder is a language model that
      has seen millions of document images during training.

    MODEL:
      microsoft/trocr-base-printed  — best for printed ID cards
      microsoft/trocr-large-printed — more accurate, slower

    INSTALL:
      pip install transformers torch pillow

    STRATEGY:
      TrOCR works on image patches, not full images.
      We crop the image into horizontal bands (each field)
      and run TrOCR on each band independently.
      This matches how the model was trained.

    RETURNS: (text: str, available: bool)
    """
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image as PILImage
        import torch
    except ImportError:
        return "", False

    try:
        info("Initializing TrOCR (microsoft/trocr-base-printed)...")
        info("First run downloads ~400MB model — cached after that")

        processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-printed",
            use_fast=True
        )
        model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-printed"
        )
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        ok(f"TrOCR loaded on {device}")

        # Convert input to PIL image
        if isinstance(image_input, str):
            pil_img = PILImage.open(image_input).convert("RGB")
        else:
            import numpy as np
            if len(image_input.shape) == 2:
                # Grayscale → RGB
                pil_img = PILImage.fromarray(
                    cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
                )
            else:
                pil_img = PILImage.fromarray(
                    cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                )

        h, w = pil_img.size[1], pil_img.size[0]

        # ── Crop into horizontal bands and OCR each ────────
        # Aadhaar card text is in rows — TrOCR reads one line at a time
        # We use 8 equal horizontal slices to cover all text rows
        num_bands = 8
        band_h    = h // num_bands
        all_lines = []

        import torch
        with torch.no_grad():
            for i in range(num_bands):
                y1 = i * band_h
                y2 = min((i + 1) * band_h, h)
                band = pil_img.crop((0, y1, w, y2))

                # Skip nearly-white bands (no text)
                import numpy as np
                band_arr = np.array(band)
                if np.mean(band_arr) > 245:
                    continue

                pixel_values = processor(
                    images=band,
                    return_tensors="pt"
                ).pixel_values.to(device)

                generated_ids = model.generate(pixel_values)
                text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0].strip()

                if text:
                    all_lines.append(text)

        combined = "\n".join(all_lines)
        ok(f"TrOCR extracted {len(combined)} chars from {len(all_lines)} bands")
        return combined, True

    except Exception as e:
        warn(f"TrOCR failed: {e}")
        return "", False


def step13_tesseract(binary, original_color, image_path=None):
    """
    Step 13 — Dual-Engine OCR: PaddleOCR (primary) + Tesseract (fallback)

    ENGINE SELECTION STRATEGY:
      1. Try PaddleOCR first — better accuracy, DL-based
      2. Always also run Tesseract — different errors, helps voting
      3. Combine all text from both engines for maximum coverage
      4. More text sources = better cross-pass voting in step 14B

    This dual-engine approach gives:
      - PaddleOCR's accuracy on clean/standard text
      - Tesseract's different error patterns for voting
      - Combined coverage catches text that either engine misses
    """
    section("13 — Dual-Engine OCR: PaddleOCR + Tesseract")

    all_texts = []
    paddle_available = False

    # ── Engine 1: PaddleOCR (primary) ────────────────────
    info("Engine 1: PaddleOCR (deep learning, primary engine)")
    info("Advantages: blur tolerant, multilingual, layout-aware")

    # Use color image for PaddleOCR — it works better on originals
    paddle_input = image_path if image_path else original_color
    paddle_text, paddle_available = run_paddleocr(paddle_input)

    if paddle_available and paddle_text.strip():
        ok(f"PaddleOCR extracted {len(paddle_text)} chars")
        all_texts.append(paddle_text)
        # Run PaddleOCR a second time on binary for extra coverage
        paddle_text2, _ = run_paddleocr(binary)
        if paddle_text2.strip():
            all_texts.append(paddle_text2)
            ok(f"PaddleOCR (binary) extracted {len(paddle_text2)} chars")
    elif not paddle_available:
        warn("PaddleOCR not installed — using Tesseract only")
        warn("To enable PaddleOCR: pip install paddlepaddle paddleocr")
        info("Tesseract will be used as the sole OCR engine.")
    else:
        warn("PaddleOCR returned no text — falling back to Tesseract only")

    # ── Engine 2: Tesseract (always runs) ────────────────
    print()
    info("Engine 2: Tesseract (classical OCR, 5-pass multi-config)")
    if paddle_available:
        info("Running in supplementary mode — adds voting diversity")
    else:
        info("Running as primary engine (PaddleOCR not available)")

    tesseract_texts = run_tesseract_passes(binary, original_color)
    all_texts.extend(tesseract_texts)

    # ── Confidence check + TrOCR retry ───────────────────
    # UPGRADE: If OCR confidence is low (very few chars extracted
    # relative to a normal Aadhaar card), trigger TrOCR as
    # Engine 3 — its transformer decoder handles blurry text better.
    #
    # HOW CONFIDENCE IS ESTIMATED:
    #   A normal Aadhaar card has ~200–600 chars of text.
    #   If all engines combined extract < 80 chars, it means
    #   the image is too blurry/noisy for standard OCR.
    #   Threshold < 80 chars → LOW CONFIDENCE → run TrOCR.
    combined_so_far = "\n".join(all_texts)
    char_count      = len(combined_so_far.replace("\n", "").strip())
    OCR_CONFIDENCE_THRESHOLD = 80   # chars — tune this if needed

    info(f"OCR confidence check: {char_count} chars extracted so far")
    if char_count < OCR_CONFIDENCE_THRESHOLD:
        warn(f"Low OCR confidence ({char_count} < {OCR_CONFIDENCE_THRESHOLD} chars)")
        warn("Triggering Engine 3: TrOCR (Transformer OCR) for blur recovery")
        print()
        trocr_input = image_path if image_path else original_color
        trocr_text, trocr_available = run_trocr(trocr_input)
        if trocr_available and trocr_text.strip():
            ok(f"TrOCR recovered {len(trocr_text)} additional chars")
            all_texts.append(trocr_text)
        elif trocr_available:
            warn("TrOCR also returned empty — image may be too degraded")
        else:
            warn("TrOCR not installed: pip install transformers torch")
    else:
        ok(f"OCR confidence OK ({char_count} chars) — TrOCR not needed")

    # ── Combine all text ──────────────────────────────────
    combined = "\n".join(all_texts)

    print("\n  ── Raw text from all engines/passes (unique lines) ──")
    seen = set()
    for line in combined.splitlines():
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            print(f"  | {line}")
    print("  ─────────────────────────────────────────────────────")

    engines_used = []
    if paddle_available:           engines_used.append("PaddleOCR")
    engines_used.append("Tesseract")
    if char_count < OCR_CONFIDENCE_THRESHOLD: engines_used.append("TrOCR")

    ok(f"OCR engines used : {' + '.join(engines_used)}")
    ok(f"Total text       : {len(combined)} chars from {len(all_texts)} passes")

    return combined, all_texts



# ═════════════════════════════════════════════════════════════
#  UPGRADE A — TrOCR (Transformer OCR)
#
#  WHY TrOCR > PaddleOCR FOR BLURRY TEXT:
#    PaddleOCR uses CRNN (CNN + RNN).
#    TrOCR uses Vision Transformer (ViT) encoder +
#    GPT-2 style text decoder.
#
#    On severely blurry images, attention mechanisms in
#    Transformers "look at the whole image context" when
#    reading each character — CRNN only looks locally.
#    This makes TrOCR significantly better on:
#      - Motion blurred photos
#      - Out-of-focus camera shots
#      - Old/faded printed cards
#
#  INSTALL:
#    pip install transformers torch Pillow
#
#  MODEL (auto-downloaded on first run ~1.3 GB):
#    microsoft/trocr-large-printed  — best for printed docs
#    microsoft/trocr-base-printed   — faster, less accurate
# ═════════════════════════════════════════════════════════════

def run_trocr(image_input, confidence_threshold=0.5):
    """
    Run Microsoft TrOCR on an image.

    HOW IT WORKS:
      1. ViT encoder splits image into 16x16 patches
         and builds a rich spatial feature representation
      2. Transformer decoder generates text token by token,
         attending to the full image at each step
      3. Returns text + per-token confidence scores

    CONFIDENCE FILTERING:
      If mean token confidence < threshold → returns empty string
      so the caller knows this result is unreliable.

    Args:
        image_input : file path (str) or numpy BGR array
        confidence_threshold : skip result if avg conf < this

    Returns:
        (text: str, confidence: float, available: bool)
    """
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image as PILImage
        import torch
    except ImportError:
        return "", 0.0, False

    try:
        info("Loading TrOCR model (microsoft/trocr-large-printed)...")
        info("First run downloads ~1.3 GB — subsequent runs use cache")

        processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-large-printed"
        )
        model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-large-printed"
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        ok(f"TrOCR loaded on {device}")

        # Convert input to PIL RGB
        if isinstance(image_input, str):
            pil_img = PILImage.open(image_input).convert("RGB")
        else:
            import cv2
            rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)

        # Preprocess and run inference
        pixel_values = processor(pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=128,
            )

        # Decode text
        generated_text = processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )[0].strip()

        # Compute mean confidence from scores
        if outputs.scores:
            import torch.nn.functional as F
            token_confs = []
            for score_tensor in outputs.scores:
                probs      = F.softmax(score_tensor, dim=-1)
                max_prob   = probs.max(dim=-1).values.item()
                token_confs.append(max_prob)
            mean_conf = sum(token_confs) / len(token_confs) if token_confs else 0.0
        else:
            mean_conf = 1.0   # no scores available → assume confident

        info(f"TrOCR result     : {generated_text!r}")
        info(f"Mean confidence  : {mean_conf:.3f}  (threshold={confidence_threshold})")

        if mean_conf < confidence_threshold:
            warn(f"TrOCR confidence {mean_conf:.3f} < {confidence_threshold} — result discarded")
            return "", mean_conf, True

        ok(f"TrOCR accepted   : {len(generated_text)} chars  (conf={mean_conf:.3f})")
        return generated_text, mean_conf, True

    except Exception as e:
        warn(f"TrOCR failed: {e}")
        return "", 0.0, False

