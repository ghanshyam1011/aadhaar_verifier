# tampering.py
# Phase 2 — Anti-Tampering Detection
# Step 20: Six independent fraud detectors run on every card.
#
# DETECTORS:
#   20A — Error Level Analysis (ELA)
#          Detects JPEG re-compression in locally edited regions.
#          A spliced name or numbercd ..\..\Downloads\aadhaar-angular-app\aadhaar-app shows as a bright patch.
#
#   20B — Noise Fingerprint Analysis
#          Camera sensor noise is spatially uniform on a genuine card.
#          A copy-pasted or Photoshopped region has different noise.
#
#   20C — Font Consistency Check
#          Genuine Aadhaar uses fixed UIDAI fonts.
#          An overlaid digit uses a different stroke width.
#
#   20D — Moiré / Screen Pattern Detection
#          A photo of a phone screen or printed photocopy shows
#          periodic interference patterns in the FFT spectrum.
#          More robust than the liveness FFT check in face_ai.py
#          because it looks at the full card, not just the face region.
#
#   20E — Hologram Region Check
#          Physical Aadhaar cards have a UIDAI hologram in the
#          top-left corner. Detects its characteristic saturation
#          pattern. Absence on a card that looks physical = suspicious.
#
#   20F — Verhoeff Checksum
#          Every valid Aadhaar number satisfies the Verhoeff
#          check-digit algorithm. A randomly fabricated number
#          fails ~90% of the time. Pure math — no image needed.
#
# RESULT:
#   step20_tampering_analysis() returns a dict:
#     {
#       'signals':     [list of fraud signal strings],
#       'score':       int 0–100  (100 = no tampering detected),
#       'verdict':     'CLEAN' / 'SUSPICIOUS' / 'TAMPERED',
#       'details':     {detector_name: {score, note, passed}},
#     }
#
# USAGE IN PIPELINE:
#   from tampering import step20_tampering_analysis
#   tamper_result = step20_tampering_analysis(
#       front_img_bgr, back_img_bgr, fields
#   )
# ─────────────────────────────────────────────────────────────

import cv2
import numpy as np
import re
import os

from utils import section, ok, info, warn, err


# ─────────────────────────────────────────────────────────────
#  20A — ERROR LEVEL ANALYSIS (ELA)
# ─────────────────────────────────────────────────────────────

def _ela_detect(img_bgr, quality=75, scale=10):
    """
    Error Level Analysis — detect locally re-compressed regions.

    HOW IT WORKS:
      1. Save the image as JPEG at a known quality (e.g. 75%).
      2. Load it back.
      3. Subtract original from re-compressed: |orig - recompressed|
      4. Scale the difference for visibility.
      5. Analyse the difference image:
           Genuine uniform region  → low, uniform error
           Spliced / edited region → high, localised error spike

    WHY IT CATCHES FAKES:
      When someone edits a JPEG (e.g. changes a name in Photoshop
      and saves again), the edited region is re-compressed from scratch.
      It has a DIFFERENT error level than the rest of the image which
      was compressed only once. The ELA difference image lights up
      brightly exactly where the edit was made.

    Args:
        img_bgr : BGR numpy array
        quality : JPEG quality for re-compression (75 is standard)
        scale   : multiplier for difference visibility

    Returns:
        (ela_img, score, note, passed)
        score: 0–100 (100 = no suspicious regions)
    """
    if img_bgr is None or img_bgr.size == 0:
        return None, 50, "No image provided", True

    import tempfile

    # Save as JPEG at known quality
    tmp_path = os.path.join(tempfile.gettempdir(), '_ela_tmp.jpg')
    try:
        cv2.imwrite(tmp_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        recompressed = cv2.imread(tmp_path)
        if recompressed is None:
            return None, 50, "ELA temp file error", True
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Resize to same shape if needed
    if recompressed.shape != img_bgr.shape:
        recompressed = cv2.resize(recompressed, (img_bgr.shape[1], img_bgr.shape[0]))

    # Compute difference and scale
    diff = cv2.absdiff(img_bgr.astype(np.float32),
                       recompressed.astype(np.float32))
    ela_img = np.clip(diff * scale, 0, 255).astype(np.uint8)

    # Analyse the ELA image
    gray_ela = cv2.cvtColor(ela_img, cv2.COLOR_BGR2GRAY)

    # Global stats
    mean_ela = np.mean(gray_ela)
    std_ela  = np.std(gray_ela)

    # Find suspicious bright spots (potential edit regions)
    # Threshold: pixels > mean + 3*std are outliers
    threshold = mean_ela + 3 * std_ela
    suspicious_mask = (gray_ela > threshold).astype(np.uint8)
    suspicious_ratio = np.sum(suspicious_mask) / suspicious_mask.size

    # Find largest suspicious contiguous region
    contours, _ = cv2.findContours(
        suspicious_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour_area = max((cv2.contourArea(c) for c in contours), default=0)
    total_area = img_bgr.shape[0] * img_bgr.shape[1]
    max_region_ratio = max_contour_area / total_area

    info(f"  ELA: mean={mean_ela:.1f} std={std_ela:.1f} "
         f"suspicious_ratio={suspicious_ratio:.3f} "
         f"max_region={max_region_ratio:.3f}")

    # Score: high score = clean
    # A suspicious region > 2% of card area is a strong tampering signal
    if max_region_ratio > 0.05:
        score = 20
        note  = (f"Large suspicious region detected ({max_region_ratio:.1%} of card). "
                 f"Possible text/number overlay at a different JPEG compression level.")
        passed = False
    elif max_region_ratio > 0.02:
        score = 55
        note  = (f"Minor ELA anomaly ({max_region_ratio:.1%} of card). "
                 f"Could be normal JPEG artefact or minor edit — review manually.")
        passed = True  # warn but don't fail
    elif suspicious_ratio > 0.08:
        score = 65
        note  = (f"Elevated ELA noise ({suspicious_ratio:.1%} of pixels). "
                 f"Possible low-quality scan artefacts.")
        passed = True
    else:
        score = 90
        note  = "ELA clean — no suspicious re-compression regions detected."
        passed = True

    ok(f"  ELA score: {score}/100 — {note[:60]}")
    return ela_img, score, note, passed


# ─────────────────────────────────────────────────────────────
#  20B — NOISE FINGERPRINT ANALYSIS
# ─────────────────────────────────────────────────────────────

def _noise_fingerprint(img_bgr):
    """
    Detect copy-paste / clone-stamp manipulation via noise analysis.

    HOW IT WORKS:
      Camera sensor noise (photon shot noise + read noise) is:
        - Spatially random (different at every pixel)
        - Statistically uniform across a genuine image
        - DIFFERENT between source and destination of a clone stamp

      We extract the noise layer using Median Filter Residual:
        noise = original - median_filter(original)

      Then we analyse the noise for spatial uniformity.
      A cloned region will have a different noise texture
      from the surrounding genuine region.

      We use LOCAL VARIANCE MAP: compute variance of noise in
      small blocks. A tampered block has significantly lower
      variance (because copy-pasted content has correlated noise).

    Returns:
        (score, note, passed)
    """
    if img_bgr is None or img_bgr.size == 0:
        return 50, "No image provided", True

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Extract noise via median filter residual
    median = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    noise  = gray - median

    # Compute local variance in 16x16 blocks
    h, w    = noise.shape
    blk     = 16
    rows    = h // blk
    cols    = w // blk
    variances = []

    for r in range(rows):
        for c in range(cols):
            patch = noise[r*blk:(r+1)*blk, c*blk:(c+1)*blk]
            variances.append(np.var(patch))

    variances = np.array(variances)
    global_mean_var = np.mean(variances)
    global_std_var  = np.std(variances)

    if global_mean_var < 1e-6:
        return 50, "Image too uniform for noise analysis", True

    # Normalised coefficient of variation
    cv_noise = global_std_var / (global_mean_var + 1e-6)

    # Detect anomalously LOW variance blocks (cloned regions)
    low_thresh  = global_mean_var - 2.5 * global_std_var
    anomalous   = np.sum(variances < low_thresh)
    anomaly_ratio = anomalous / len(variances) if len(variances) > 0 else 0

    info(f"  Noise: mean_var={global_mean_var:.3f} cv={cv_noise:.3f} "
         f"anomalous_blocks={anomalous}/{len(variances)} "
         f"({anomaly_ratio:.1%})")

    if anomaly_ratio > 0.12:
        score  = 25
        note   = (f"Noise fingerprint anomaly: {anomaly_ratio:.1%} of blocks have "
                  f"abnormally low noise — possible cloned/pasted region.")
        passed = False
    elif anomaly_ratio > 0.06:
        score  = 60
        note   = (f"Minor noise anomaly ({anomaly_ratio:.1%} abnormal blocks). "
                  f"May be compression or low-quality scan.")
        passed = True
    else:
        score  = 88
        note   = "Noise fingerprint uniform — no clone-stamp detected."
        passed = True

    ok(f"  Noise score: {score}/100 — {note[:60]}")
    return score, note, passed


# ─────────────────────────────────────────────────────────────
#  20C — FONT CONSISTENCY CHECK
# ─────────────────────────────────────────────────────────────

def _font_consistency(img_bgr, aadhaar_number=None):
    """
    Check that text in the Aadhaar number region uses consistent
    stroke width — a sign that it was printed in the original font.

    HOW IT WORKS:
      Genuine UIDAI Aadhaar numbers are printed in a specific font
      with a specific stroke width. When someone overlays a fake
      number (e.g. in Photoshop), the stroke width is usually
      different — either because:
        1. They used a different font
        2. They rasterised at a different resolution
        3. They used a screen font vs a print font

      We measure stroke width using the Distance Transform:
        - Binarise the Aadhaar number region
        - Apply distance transform (each fg pixel → distance to bg)
        - The peak of the distance transform histogram = stroke half-width
        - Compare to expected range for genuine Aadhaar

      Expected stroke width for Aadhaar numbers at 1600px width:
        Min: 2px  Max: 8px  (depends on card DPI and scan quality)

    Returns:
        (score, note, passed)
    """
    if img_bgr is None or img_bgr.size == 0:
        return 50, "No image for font check", True

    h, w = img_bgr.shape[:2]

    # Aadhaar number is printed at the BOTTOM of the front side
    # Crop the bottom 25% of the card — that's where the number lives
    bottom_region = img_bgr[int(h * 0.70):h, int(w * 0.20):int(w * 0.85)]

    if bottom_region.size == 0:
        return 50, "Could not crop Aadhaar number region", True

    gray   = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
    _, bw  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Distance transform on text pixels
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)

    # Histogram of non-zero distance values (= stroke half-widths)
    nonzero_dist = dist[dist > 0]
    if len(nonzero_dist) < 100:
        return 50, "Too few text pixels in Aadhaar number region", True

    median_stroke = np.median(nonzero_dist)
    std_stroke    = np.std(nonzero_dist)
    cv_stroke     = std_stroke / (median_stroke + 1e-6)

    info(f"  Font: median_stroke={median_stroke:.2f}px "
         f"std={std_stroke:.2f}px cv={cv_stroke:.3f}")

    # High coefficient of variation in stroke width = mixed fonts
    # Expected: genuine Aadhaar has cv_stroke < 0.60
    if cv_stroke > 1.2:
        score  = 30
        note   = (f"High font stroke variation (cv={cv_stroke:.2f}). "
                  f"Mixed character fonts detected — possible text overlay.")
        passed = False
    elif cv_stroke > 0.80:
        score  = 65
        note   = (f"Moderate font variation (cv={cv_stroke:.2f}). "
                  f"Could be scan quality or bold/regular mix.")
        passed = True
    else:
        score  = 90
        note   = (f"Font stroke consistent (cv={cv_stroke:.2f}, "
                  f"median={median_stroke:.1f}px).")
        passed = True

    ok(f"  Font score: {score}/100 — {note[:60]}")
    return score, note, passed


# ─────────────────────────────────────────────────────────────
#  20D — MOIRE / SCREEN PATTERN DETECTION
# ─────────────────────────────────────────────────────────────

def _moire_detect(img_bgr):
    """
    Detect Moiré patterns — signature of a photographed screen
    or photocopied document.

    HOW IT WORKS:
      A physical Aadhaar card photographed directly:
        → Natural grain texture in mid-frequency FFT bands
        → No periodic peaks in high-frequency bands

      A photo of a phone screen showing the Aadhaar:
        → Screen pixel grid creates strong periodic peaks at
          frequencies corresponding to the screen resolution
        → 60/90/120Hz refresh rate is visible as horizontal bands

      A photocopied Aadhaar:
        → Halftone dot pattern creates peaks at ~45° in FFT
        → Regular dot-matrix spacing appears as multiple peaks

      DETECTION METHOD:
        1. Compute 2D FFT of the full card image
        2. Shift DC to centre (fftshift)
        3. Mask out the DC component and low-frequency content
        4. Count how many sharp peaks exist in high-frequency ring
        5. Real card: few peaks, moderate amplitude
           Screen/photocopy: many peaks, high amplitude at regular intervals

    Returns:
        (score, note, passed)
    """
    if img_bgr is None or img_bgr.size == 0:
        return 50, "No image for Moiré detection", True

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Resize to standard size for consistent FFT
    std_size = (512, 512)
    gray_std = cv2.resize(gray, std_size, interpolation=cv2.INTER_AREA)

    # 2D FFT
    fft    = np.fft.fft2(gray_std.astype(np.float64))
    fft_sh = np.fft.fftshift(fft)
    mag    = np.abs(fft_sh)

    h, w  = mag.shape
    cy, cx = h // 2, w // 2

    # Mask out DC component (centre 15x15)
    mag_work = mag.copy()
    mag_work[cy-7:cy+8, cx-7:cx+8] = 0

    # Log magnitude for better dynamic range
    log_mag = np.log1p(mag_work)

    # Analyse high-frequency ring (40-80% of max radius)
    # This is where screen/halftone patterns appear
    r_min = int(0.40 * min(h, w) / 2)
    r_max = int(0.80 * min(h, w) / 2)

    hf_mask = np.zeros_like(log_mag)
    for y in range(h):
        for x in range(w):
            r = np.sqrt((y - cy)**2 + (x - cx)**2)
            if r_min <= r <= r_max:
                hf_mask[y, x] = 1

    hf_region = log_mag * hf_mask
    hf_mean   = np.mean(hf_region[hf_mask > 0]) if np.any(hf_mask > 0) else 0
    hf_max    = np.max(hf_region)

    # Count sharp peaks: pixels > hf_mean + 3*std in high-freq ring
    hf_vals = hf_region[hf_mask > 0]
    if len(hf_vals) == 0:
        return 50, "FFT analysis failed", True

    hf_std    = np.std(hf_vals)
    peak_thresh = hf_mean + 3.5 * hf_std
    peak_count  = np.sum(hf_region > peak_thresh)
    peak_ratio  = peak_count / np.sum(hf_mask)

    # Global peak ratio (max vs mean across whole spectrum)
    global_ratio = hf_max / (np.mean(log_mag[log_mag > 0]) + 1e-6)

    info(f"  Moiré: hf_mean={hf_mean:.2f} hf_std={hf_std:.2f} "
         f"peaks={peak_count} peak_ratio={peak_ratio:.4f} "
         f"global_ratio={global_ratio:.1f}")

    # Scoring
    if peak_ratio > 0.008 and global_ratio > 12:
        score  = 15
        note   = (f"Strong periodic pattern in FFT ({peak_count} peaks, "
                  f"ratio={global_ratio:.1f}). Card is likely a photograph "
                  f"of a screen or a photocopy. NOT the physical card.")
        passed = False
    elif peak_ratio > 0.004 or global_ratio > 8:
        score  = 55
        note   = (f"Mild periodic pattern ({peak_count} peaks). "
                  f"Possible scan artefact or low-quality photocopy.")
        passed = True
    else:
        score  = 92
        note   = "No Moiré pattern detected — card appears to be original."
        passed = True

    ok(f"  Moiré score: {score}/100 — {note[:60]}")
    return score, note, passed


# ─────────────────────────────────────────────────────────────
#  20E — HOLOGRAM REGION CHECK
# ─────────────────────────────────────────────────────────────

def _hologram_check(img_bgr):
    """
    Detect UIDAI hologram presence in the top-left of the card.

    HOW IT WORKS:
      Physical Aadhaar cards have a UIDAI hologram sticker in
      the top-left quadrant (roughly x=0–20%, y=0–30%).
      This hologram has a characteristic rainbow saturation pattern:
        - High saturation (S channel in HSV > 80)
        - Colour varies rapidly across a small region
        - High hue variance in a small area

      A printed/digital/photocopy card will NOT have this hologram.
      A hologram photographed under direct light shows up clearly.
      A hologram photographed under indirect light shows up as
      an area of unusually high colour variance.

      NOTE: This check is only meaningful for physical cards.
      e-Aadhaar (PDF download) will not have a hologram — we
      skip this check if card_type indicates e-Aadhaar.

    Returns:
        (score, note, passed, hologram_found)
    """
    if img_bgr is None or img_bgr.size == 0:
        return 50, "No image for hologram check", True, False

    h, w = img_bgr.shape[:2]

    # Crop top-left quadrant where hologram lives
    holo_region = img_bgr[0:int(h * 0.30), 0:int(w * 0.22)]
    if holo_region.size == 0:
        return 50, "Could not crop hologram region", True, False

    # Convert to HSV
    hsv = cv2.cvtColor(holo_region, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1].astype(np.float32)
    h_channel = hsv[:, :, 0].astype(np.float32)

    # Hologram indicators:
    #   1. High saturation pixels (S > 60)
    #   2. High hue variance (rainbow = many different hues close together)
    high_sat_mask = (s_channel > 60).astype(np.uint8)
    high_sat_ratio = np.sum(high_sat_mask) / high_sat_mask.size

    # Hue variance in high-saturation region
    if np.sum(high_sat_mask) > 50:
        hue_in_holo = h_channel[high_sat_mask > 0]
        hue_std     = np.std(hue_in_holo)
    else:
        hue_std = 0

    # Mean saturation
    mean_sat = np.mean(s_channel)

    info(f"  Hologram: mean_sat={mean_sat:.1f} "
         f"high_sat_ratio={high_sat_ratio:.3f} "
         f"hue_std={hue_std:.1f}")

    hologram_found = False

    # Hologram present: high saturation + high hue variance
    if high_sat_ratio > 0.10 and hue_std > 15:
        hologram_found = True
        score  = 92
        note   = (f"UIDAI hologram pattern detected "
                  f"(sat_ratio={high_sat_ratio:.1%}, hue_var={hue_std:.0f}).")
        passed = True

    # Some saturation but weak — indirect lighting
    elif high_sat_ratio > 0.04 or mean_sat > 25:
        hologram_found = True  # give benefit of doubt
        score  = 75
        note   = (f"Possible hologram region (low confidence). "
                  f"Try photographing card under different lighting.")
        passed = True

    # No hologram signal — suspicious for physical card
    else:
        score  = 45
        note   = (f"No hologram pattern detected in top-left region "
                  f"(sat={mean_sat:.0f}). If this is a physical card, "
                  f"the hologram may be absent — check manually.")
        passed = True  # warn but don't fail (e-Aadhaar legitimately has none)

    ok(f"  Hologram score: {score}/100 — {note[:60]}")
    return score, note, passed, hologram_found


# ─────────────────────────────────────────────────────────────
#  20F — VERHOEFF CHECKSUM
# ─────────────────────────────────────────────────────────────

# Verhoeff algorithm tables — mathematically derived, not configurable
_VERHOEFF_D = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6],
    [3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8],
    [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],
    [7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4],
    [9,8,7,6,5,4,3,2,1,0],
]
_VERHOEFF_P = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2],
    [8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0],
    [4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],
    [7,0,4,6,9,1,3,2,5,8],
]
_VERHOEFF_INV = [0,4,3,2,1,9,8,7,6,5]


def _verhoeff_validate(number_str):
    """
    Validate an Aadhaar number using the Verhoeff check-digit algorithm.

    WHAT IS VERHOEFF:
      The Verhoeff algorithm is a check-digit scheme based on:
        - The dihedral group D5 (multiplication table _D)
        - A permutation function (_P) applied to each digit position
        - An inverse table (_INV) for verification

      It detects ALL single-digit errors and ALL adjacent
      transpositions — much stronger than Luhn (used in credit cards).

      UIDAI uses Verhoeff for Aadhaar number validation.
      A valid Aadhaar number, when run through Verhoeff, gives 0.
      Any randomly fabricated 12-digit number fails ~90% of the time.

    Args:
        number_str: Aadhaar number as string (spaces stripped)

    Returns:
        (is_valid: bool, note: str)
    """
    digits = re.sub(r'\D', '', str(number_str or ''))

    if len(digits) != 12:
        return False, f"Need 12 digits for Verhoeff, got {len(digits)}"

    if digits[0] in '01':
        return False, "Aadhaar cannot start with 0 or 1"

    try:
        c = 0
        # Process digits right to left
        for i, d in enumerate(reversed(digits)):
            p = _VERHOEFF_P[i % 8][int(d)]
            c = _VERHOEFF_D[c][p]

        if c == 0:
            return True, "Verhoeff checksum valid ✓"
        else:
            return False, (f"Verhoeff checksum FAILED (result={c}, expected 0). "
                           f"This Aadhaar number is mathematically invalid — "
                           f"likely fabricated or OCR-corrupted.")
    except (IndexError, ValueError) as e:
        return False, f"Verhoeff computation error: {e}"


def _verhoeff_check(aadhaar_number):
    """
    Run Verhoeff on the extracted Aadhaar number.

    Returns:
        (score, note, passed)
    """
    if not aadhaar_number:
        return 40, "No Aadhaar number to validate", False

    is_valid, note = _verhoeff_validate(aadhaar_number)

    if is_valid:
        score  = 100
        passed = True
        ok(f"  Verhoeff: PASS — {note}")
    else:
        score  = 0
        passed = False
        warn(f"  Verhoeff: FAIL — {note}")

    return score, note, passed


# ─────────────────────────────────────────────────────────────
#  MAIN STEP — Run all 6 detectors
# ─────────────────────────────────────────────────────────────

def step20_tampering_analysis(front_img_bgr, back_img_bgr, fields,
                               card_type=None):
    """
    Step 20 — Anti-Tampering Analysis.

    Runs all 6 detectors and returns a consolidated result.

    Args:
        front_img_bgr : BGR numpy array of front card image
        back_img_bgr  : BGR numpy array of back card image (may be None)
        fields        : dict from step14_extract / step14b_correct
        card_type     : str — 'Front Side', 'Back Side', 'e-Aadhaar...', etc.

    Returns:
        dict with keys:
          signals   : list of fraud signal strings (plain English)
          score     : int 0–100  (100 = no tampering)
          verdict   : 'CLEAN' / 'SUSPICIOUS' / 'TAMPERED'
          details   : {detector: {score, note, passed}}
          passed    : bool — True if no hard fails
    """
    section("20 — Anti-Tampering Analysis")
    info("Running 6 independent fraud detectors...")
    print()

    details  = {}
    signals  = []
    is_edigital = bool(card_type and 'e-Aadhaar' in str(card_type))

    # ── 20A: ELA ─────────────────────────────────────────────
    info("20A — Error Level Analysis (ELA)")
    ela_img, ela_score, ela_note, ela_passed = _ela_detect(front_img_bgr)
    details['ela'] = {'score': ela_score, 'note': ela_note, 'passed': ela_passed}
    if not ela_passed:
        signals.append(f"ELA: {ela_note}")

    # ── 20B: Noise Fingerprint ────────────────────────────────
    print()
    info("20B — Noise Fingerprint Analysis")
    nf_score, nf_note, nf_passed = _noise_fingerprint(front_img_bgr)
    details['noise'] = {'score': nf_score, 'note': nf_note, 'passed': nf_passed}
    if not nf_passed:
        signals.append(f"Noise: {nf_note}")

    # ── 20C: Font Consistency ─────────────────────────────────
    print()
    info("20C — Font Consistency Check")
    fc_score, fc_note, fc_passed = _font_consistency(
        front_img_bgr, fields.get('aadhaar_number'))
    details['font'] = {'score': fc_score, 'note': fc_note, 'passed': fc_passed}
    if not fc_passed:
        signals.append(f"Font: {fc_note}")

    # ── 20D: Moiré ───────────────────────────────────────────
    print()
    info("20D — Moiré / Screen Pattern Detection")
    # Use back image if available (larger area, clearer pattern)
    moire_img = back_img_bgr if back_img_bgr is not None else front_img_bgr
    mr_score, mr_note, mr_passed = _moire_detect(moire_img)
    details['moire'] = {'score': mr_score, 'note': mr_note, 'passed': mr_passed}
    if not mr_passed:
        signals.append(f"Moiré: {mr_note}")

    # ── 20E: Hologram ─────────────────────────────────────────
    print()
    info("20E — Hologram Region Check")
    if is_edigital:
        holo_score  = 80
        holo_note   = "e-Aadhaar / digital card — hologram not expected. Skipped."
        holo_passed = True
        holo_found  = False
        info(f"  Hologram check skipped for e-Aadhaar (card_type='{card_type}')")
    else:
        holo_score, holo_note, holo_passed, holo_found = _hologram_check(front_img_bgr)
    details['hologram'] = {
        'score': holo_score, 'note': holo_note,
        'passed': holo_passed, 'found': holo_found
    }
    if not holo_passed:
        signals.append(f"Hologram: {holo_note}")

    # ── 20F: Verhoeff ─────────────────────────────────────────
    print()
    info("20F — Verhoeff Checksum Validation")
    vf_score, vf_note, vf_passed = _verhoeff_check(fields.get('aadhaar_number'))
    details['verhoeff'] = {'score': vf_score, 'note': vf_note, 'passed': vf_passed}
    if not vf_passed:
        signals.append(f"Verhoeff checksum: {vf_note}")

    # ── Compute overall score ─────────────────────────────────
    # Weighted average — Verhoeff and ELA are highest weight
    weights = {
        'ela':      0.25,   # strongest visual tamper signal
        'verhoeff': 0.25,   # strongest mathematical signal
        'noise':    0.20,   # good clone detector
        'moire':    0.15,   # catches screen/photocopy fraud
        'font':     0.10,   # catches font overlay
        'hologram': 0.05,   # supplementary
    }
    scores = {
        'ela':      ela_score,
        'verhoeff': vf_score,
        'noise':    nf_score,
        'moire':    mr_score,
        'font':     fc_score,
        'hologram': holo_score,
    }
    overall = int(sum(scores[k] * weights[k] for k in weights))

    # Hard fails override the score
    hard_fails = [k for k in ['verhoeff', 'ela', 'moire']
                  if not details[k]['passed']]

    if hard_fails:
        overall = min(overall, 35)

    # Verdict
    if overall >= 75 and not hard_fails:
        verdict = "CLEAN ✓"
    elif overall >= 50 or not hard_fails:
        verdict = "SUSPICIOUS ⚠"
    else:
        verdict = "TAMPERED ✗"

    # Summary printout
    print()
    print(f"  {'─'*56}")
    print(f"  {'Detector':<22} {'Score':>6}   {'Status'}")
    print(f"  {'─'*56}")
    detector_labels = {
        'ela':      '20A ELA',
        'noise':    '20B Noise Fingerprint',
        'font':     '20C Font Consistency',
        'moire':    '20D Moiré Detection',
        'hologram': '20E Hologram Check',
        'verhoeff': '20F Verhoeff Checksum',
    }
    for key, label in detector_labels.items():
        d     = details[key]
        sym   = "✓" if d['passed'] else "✗"
        print(f"  {label:<22} {d['score']:>5}/100  {sym}  {d['note'][:35]}")

    print(f"  {'─'*56}")
    print(f"  Overall Tampering Score : {overall}/100")
    print(f"  Verdict                 : {verdict}")
    if signals:
        print(f"  Fraud Signals ({len(signals)}):")
        for s in signals:
            print(f"    ✗ {s[:70]}")
    print(f"  {'─'*56}")

    if overall >= 75 and not signals:
        ok(f"Tampering analysis complete: {verdict} ({overall}/100)")
    else:
        warn(f"Tampering analysis: {verdict} ({overall}/100) — "
             f"{len(signals)} signal(s)")

    return {
        'signals': signals,
        'score':   overall,
        'verdict': verdict,
        'details': details,
        'passed':  len(hard_fails) == 0,
    }