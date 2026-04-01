# llm_correction.py
# Step 14C: Groq LLM correction + local flan-t5 fallback
# Also contains: ocr_confidence_score(), run_with_confidence_gate()
# ─────────────────────────────────────────────────────────────
# No code changed — same functions, split into own file.

import re
import os

from utils import section, ok, info, warn, err
from preprocessing import (
    step2b_super_resolution, step3_orient, step4_mask_photo,
    step5_remove_color_noise, step6_grayscale, step7_denoise,
    step8_clahe, step9_adaptive_sharpen, step10_binarize,
    step11_deskew, step12_morph
)
from ocr_engines import step13_tesseract


# ═════════════════════════════════════════════════════════════
#  UPGRADE B — LLM Correction via Groq API
#
#  WHY GROQ:
#    Groq provides a free-tier API that runs open-source LLMs
#    (LLaMA 3, Mixtral) at very high speed on custom hardware.
#    It uses the OpenAI-compatible chat completions format,
#    so pip install openai is all you need.
#
#  SETUP (one time):
#    1. Get free API key: https://console.groq.com
#    2. pip install openai
#    3. Add GROQ_API_KEY=gsk_... to your .env file
#
#  MODEL USED: llama-3.3-70b-versatile
#    — Best open-source model for Indian name correction
#    — Fast on Groq hardware (~1–2 sec per correction)
#    — Free tier: 14,400 requests/day
#
#  GRACEFUL FALLBACK:
#    If GROQ_API_KEY not set → falls back to local flan-t5-small
#    If both unavailable → returns fields unchanged (never crashes)
# ═════════════════════════════════════════════════════════════

def llm_correct_fields(raw_ocr_text, fields, doc_type="aadhaar", api_key=None):
    """
    Use Groq LLM to correct OCR extraction errors.
    Single unified function — always uses Groq.

    Args:
        raw_ocr_text : combined text from all OCR passes
        fields       : dict of already-extracted fields
        doc_type     : "aadhaar" or "pan"
        api_key      : Groq API key (falls back to GROQ_API_KEY env var)

    Returns:
        corrected_fields dict (same keys as input fields)
    """
    section("14C — Groq LLM Correction")

    import os as _os
    key = api_key or _os.environ.get("GROQ_API_KEY", "")

    # Build field summary for prompt
    field_lines = "\n".join(
        f"  {k}: {v}" for k, v in fields.items()
        if k not in ("issued_by",) and v
        and not k.startswith("address_")   # skip raw address noise
    )

    if doc_type == "pan":
        doc_desc   = "Indian PAN card"
        field_spec = (
            '"pan_number": "AAAAA9999A format",\n'
            '  "name": "FULL NAME IN CAPS",\n'
            '  "father_name": "FATHER NAME IN CAPS",\n'
            '  "dob": "dd/mm/yyyy"'
        )
    else:
        doc_desc   = "Indian Aadhaar card"
        field_spec = (
            '"name": "FULL NAME IN CAPS",\n'
            '  "dob": "dd/mm/yyyy",\n'
            '  "gender": "Male or Female or Transgender",\n'
            '  "aadhaar_number": "XXXX XXXX XXXX"\n'
            '  "father_husband_name": "NAME IN CAPS or null"'
        )

    prompt = f"""You are an OCR correction expert for {doc_desc} documents in India.

The system has already extracted these fields from the Aadhaar card:
{field_lines}

Your job: fix ONLY clear character-level OCR misreads in the extracted values.
Rules:
1. If a field value looks correct — return it UNCHANGED. Do not "improve" it.
2. Fix ONLY single-character confusions: 0↔O, 1↔I/l, 5↔S, 8↔B, rn↔m
3. For names: GHANSHYAM, JETHARAM, KUMAVAT are valid Indian names — keep them
4. Aadhaar number must be exactly 12 digits in format XXXX XXXX XXXX
5. Date must be dd/mm/yyyy with day 1-31, month 1-12, year 1900-2025
6. Gender: normalize to exactly Male / Female / Transgender
7. DO NOT invent or guess any field. Use null for fields you are unsure about.
8. DO NOT change a name that is already a valid Indian name.

Return ONLY valid JSON, no explanation, no markdown fences:
{{
  {field_spec}
}}"""

    # ── Groq API ──────────────────────────────────────────────
    if key:
        info("LLM correction: Groq API (llama-3.3-70b-versatile)")
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key  = key,
                base_url = "https://api.groq.com/openai/v1",
            )
            response = client.chat.completions.create(
                model       = "llama-3.3-70b-versatile",
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 300,
                temperature = 0.1,   # deterministic corrections
            )
            raw_resp = response.choices[0].message.content.strip()
            ok(f"Groq responded ({len(raw_resp)} chars)")

            import json as _json, re as _re
            # Strip any accidental markdown fences
            clean = _re.sub(r'```(?:json)?', '', raw_resp).strip().rstrip('`').strip()
            # Extract the JSON object
            json_match = _re.search(r'\{.*\}', clean, _re.DOTALL)
            if not json_match:
                warn("Groq response had no JSON object — skipping")
                return fields

            corrected_vals = _json.loads(json_match.group())
            corrections_made = 0
            for k, new_val in corrected_vals.items():
                if not new_val or str(new_val).lower() in ('null', 'none', ''):
                    continue
                old_val = fields.get(k)

                # ── Name sanity check ─────────────────────────────
                # Reject the LLM's name suggestion if ANY word initial
                # differs from the current name. Indian names have
                # predictable initials — changing them means hallucination.
                if k == 'name' and old_val and new_val:
                    old_words = str(old_val).strip().split()
                    new_words = str(new_val).strip().split()
                    if len(old_words) >= 2 and len(new_words) >= 2:
                        # Must have same word count
                        if len(old_words) != len(new_words):
                            info(f"  Groq name '{new_val}' rejected — "
                                 f"word count changed {len(old_words)}→{len(new_words)}")
                            continue
                        old_initials = [w[0].upper() for w in old_words]
                        new_initials = [w[0].upper() for w in new_words]
                        diff_initials = sum(
                            1 for a, b in zip(old_initials, new_initials) if a != b
                        )
                        # Zero tolerance: no initial should change
                        if diff_initials > 0:
                            info(f"  Groq name '{new_val}' rejected — "
                                 f"initials differ from '{old_val}' (keeping original)")
                            continue

                if old_val and str(new_val) != str(old_val):
                    warn(f"  Groq corrected {k}: '{old_val}'  →  '{new_val}'")
                    fields[k] = new_val
                    corrections_made += 1
                elif not old_val and new_val:
                    ok(f"  Groq recovered {k}: '{new_val}'")
                    fields[k] = new_val
                    corrections_made += 1

            if corrections_made == 0:
                ok("Groq: all fields look correct — no changes needed")
            else:
                ok(f"Groq: made {corrections_made} correction(s)")

            return fields

        except ImportError:
            warn("openai package not installed — pip install openai")
            warn("Falling back to local LLM...")
        except Exception as e:
            warn(f"Groq API call failed: {e}")
            warn("Falling back to local LLM...")
    else:
        warn("GROQ_API_KEY not set — skipping Groq correction")
        info("To enable: Add GROQ_API_KEY=gsk_... to your .env file  (free at console.groq.com)")

    # ── Local fallback: flan-t5-small ─────────────────────────
    info("Trying local LLM fallback (google/flan-t5-small, ~80MB)...")
    try:
        from transformers import pipeline as hf_pipeline
        corrector = hf_pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device=-1,
            max_new_tokens=60,
        )
        ok("Local flan-t5-small loaded")

        # Name correction
        name = fields.get("name")
        if name:
            p = (f"Fix OCR errors in this Indian person name. "
                 f"Return only the corrected name, nothing else. Name: {name}")
            result = corrector(p)[0]["generated_text"].strip()
            if (result and len(result) > 3
                    and result.replace(" ", "").replace(".", "").isalpha()
                    and len(result.split()) >= 2
                    and result.upper() != name.upper()):
                warn(f"  Local LLM name fix: '{name}' → '{result.upper()}'")
                fields["name"] = result.upper()

        # DOB correction (only if obviously broken)
        dob = fields.get("dob")
        if dob and ("?" in dob or len(dob) < 8):
            p = (f"Fix this Aadhaar card date of birth to DD/MM/YYYY format. "
                 f"Return only the date. Date: {dob}")
            result = corrector(p)[0]["generated_text"].strip()
            import re as _re2
            if _re2.match(r'\d{2}/\d{2}/\d{4}', result):
                warn(f"  Local LLM DOB fix: '{dob}' → '{result}'")
                fields["dob"] = result

        ok("Local LLM correction complete")
        return fields

    except ImportError:
        warn("transformers not installed — pip install transformers torch sentencepiece")
    except Exception as e:
        warn(f"Local LLM failed: {e}")

    warn("All LLM correction skipped — fields returned as-is from regex stage")
    return fields


# ═════════════════════════════════════════════════════════════
#  UPGRADE C — Confidence-Gated Super Resolution
#
#  THE PROBLEM WITH STATIC SUPER RESOLUTION:
#    Running SR on every image wastes time on already-sharp
#    images and can even over-sharpen them (halos, artifacts).
#
#  THE SOLUTION — Adaptive confidence loop:
#    1. Measure initial sharpness (Laplacian variance)
#    2. Run OCR, measure how many fields were found
#    3. If confidence < threshold AND image was blurry:
#       → Run super resolution
#       → Re-run OCR
#       → Keep whichever result has more fields found
#
#  This is the same logic used in Google Lens:
#    "If we're not confident, try harder preprocessing."
#
# ═════════════════════════════════════════════════════════════

def ocr_confidence_score(fields):
    """
    Score how complete an OCR result is.
    Used to decide whether to retry with super resolution.

    Scoring:
      Each found field = +1 point
      name found       = +2 (most important)
      key number found = +3 (Aadhaar / PAN number — hardest to get)

    Returns float 0.0 to 1.0
    """
    score   = 0
    max_pts = 0

    checks = [
        ("name",           2),
        ("dob",            1),
        ("gender",         1),
        ("aadhaar_number", 3),   # Aadhaar
        ("pan_number",     3),   # PAN
        ("father_name",    1),
    ]
    for key, pts in checks:
        max_pts += pts
        if fields.get(key):
            score += pts

    return round(score / max_pts, 3) if max_pts > 0 else 0.0


def run_with_confidence_gate(
        img, image_path, extract_fn, correct_fn,
        blur_score, confidence_threshold=0.6):
    """
    Confidence-gated OCR loop.

    FLOW:
      1. Run standard OCR pipeline → get fields
      2. Score how many fields were found (0.0 – 1.0)
      3. If score < threshold AND image was blurry:
           a. Run Real-ESRGAN super resolution
           b. Re-run full preprocessing + OCR
           c. Score the new result
           d. Keep whichever scored higher
      4. Return best fields + final confidence score

    This means super resolution only runs when needed,
    not on every image — saving time on clear photos.

    Args:
        img                  : preprocessed binary image
        image_path           : path to original color image
        extract_fn           : step14_extract function
        correct_fn           : step14b_correct function
        blur_score           : Laplacian variance from step 9
        confidence_threshold : retry SR if score below this

    Returns:
        (fields, confidence_score)
    """
    section("13D — Confidence-Gated Processing")

    # ── Pass 1: Standard OCR ─────────────────────────────
    info("Pass 1: Running standard OCR pipeline...")
    import cv2 as _cv2

    # Load original color image for OCR
    original_color = _cv2.imread(image_path) if isinstance(image_path, str) else image_path

    binary_pass1 = img  # already preprocessed binary from main pipeline

    combined1, all_texts1 = step13_tesseract(binary_pass1, original_color, image_path)
    fields1    = extract_fn(combined1)
    fields1    = correct_fn(fields1, all_texts1)
    conf1      = ocr_confidence_score(fields1)

    ok(f"Pass 1 confidence score : {conf1:.3f}")
    for k, v in fields1.items():
        if k not in ("issued_by",) and v:
            ok(f"  {k:<20}: {v}")

    # ── Decide whether to retry with SR ──────────────────
    image_was_blurry = blur_score < 200

    if conf1 >= confidence_threshold:
        ok(f"Confidence {conf1:.3f} >= {confidence_threshold} — no retry needed")
        return fields1, conf1

    if not image_was_blurry:
        info(f"Confidence {conf1:.3f} < {confidence_threshold} but image is sharp")
        info("Low confidence is likely due to card quality, not blur")
        info("SR retry skipped — would not help here")
        return fields1, conf1

    # ── Pass 2: SR-enhanced OCR ──────────────────────────
    warn(f"Confidence {conf1:.3f} < {confidence_threshold} AND image was blurry")
    warn("Retrying with Super Resolution...")

    sr_img = step2b_super_resolution(original_color)
    if sr_img is original_color:
        warn("SR not available — cannot retry")
        return fields1, conf1

    # Rerun preprocessing on SR image
    info("Rerunning preprocessing on SR-enhanced image...")
    oriented2  = step3_orient(sr_img)
    masked2    = step4_mask_photo(oriented2)
    no_color2  = step5_remove_color_noise(masked2)
    gray2      = step6_grayscale(no_color2)
    denoised2  = step7_denoise(gray2)
    enhanced2  = step8_clahe(denoised2)
    sharpened2 = step9_adaptive_sharpen(enhanced2)
    _, _, blended2 = step10_binarize(sharpened2)
    deskewed2  = step11_deskew(blended2)
    cleaned2   = step12_morph(deskewed2)

    # Save SR image temporarily for PaddleOCR
    import tempfile, os as _os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    _cv2.imwrite(tmp_path, sr_img)

    combined2, all_texts2 = step13_tesseract(cleaned2, oriented2, tmp_path)
    _os.unlink(tmp_path)  # clean up temp file

    fields2 = extract_fn(combined2)
    fields2 = correct_fn(fields2, all_texts2)
    conf2   = ocr_confidence_score(fields2)

    ok(f"Pass 2 (SR) confidence score : {conf2:.3f}")

    # ── Pick the better result ────────────────────────────
    if conf2 > conf1:
        ok(f"SR improved confidence: {conf1:.3f}  →  {conf2:.3f}")
        ok("Using SR-enhanced result")
        return fields2, conf2
    else:
        info(f"SR did not improve confidence: {conf1:.3f} vs {conf2:.3f}")
        info("Keeping original result")
        return fields1, conf1

