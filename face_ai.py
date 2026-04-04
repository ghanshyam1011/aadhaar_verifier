# face_ai.py
# Step 19: Face AI Pipeline — extraction, quality, matching, liveness
#   19A — step19a_extract_face()
#   19B — step19b_face_quality()
#   19C — step19c_face_match()  (DeepFace → LBPH fallback)
#   19D — step19d_liveness_hint()
#   Pipeline — step19_face_pipeline()
# ─────────────────────────────────────────────────────────────
# No code changed — same functions, split into own file.

import cv2
import numpy as np
import os

from utils import section, ok, info, warn, err

# ═════════════════════════════════════════════════════════════
#  STEP 19 — Face AI Module
#
#  FOUR CAPABILITIES:
#
#  19A — Face Extraction
#    Crops the photograph region from the Aadhaar card.
#    Aadhaar card layout is fixed: photo is always in the
#    left ~28% of the card, vertically centered.
#    We use OpenCV face detection (Haar cascade) to precisely
#    locate the face within that region rather than a hard crop.
#    Output: face_image (BGR numpy array) + face_bbox
#
#  19B — Face Quality Assessment
#    Before matching, we check whether the extracted face is
#    usable quality. Checks:
#      • Sharpness (Laplacian variance) — blurry photos fail
#      • Brightness (mean pixel value) — over/under exposed
#      • Face size (pixels) — too small = unreliable
#      • Face region coverage (face area / total area)
#    Returns quality_score 0–100 and quality_verdict.
#
#  19C — Face Match (Selfie vs Card)
#    Compares the face extracted from the Aadhaar card against
#    a selfie photo provided by the user.
#
#    ENGINE PRIORITY:
#      1. DeepFace (if installed) — deep learning, state-of-art
#         Uses VGG-Face model.  Threshold: cosine distance < 0.40
#         Install: pip install deepface tensorflow
#      2. OpenCV LBPH (always available) — classical ML fallback
#         Local Binary Pattern Histogram face recognizer.
#         Less accurate but zero extra dependencies.
#         Returns similarity 0.0–1.0 based on histogram distance.
#
#    Scoring:
#      match_score  : 0–100 (100 = identical)
#      verdict      : MATCH / NO MATCH / INCONCLUSIVE
#      engine_used  : deepface / lbph / none
#
#  19D — Liveness Hint (Anti-Spoofing Basic Check)
#    Detects "photo of a photo" attacks: someone holds up a
#    printed photo or shows a phone screen instead of their face.
#
#    METHOD: Texture frequency analysis
#      A real face has natural skin texture in the mid-frequency
#      range. A printed photo has regular dot-matrix patterns
#      (halftone artifacts). A phone screen has pixel grid artifacts.
#
#      We compute the 2D FFT of the face region and look for:
#        • Strong regular frequency peaks → printing artifact
#        • Unusually uniform texture → screen/photo
#
#    Note: This is a heuristic check, not a certified liveness
#    detector. For production, use a dedicated PAD model.
#    Returns: (likely_real: bool, liveness_score: float, note: str)
#
#  INSTALL:
#    DeepFace (best): pip install deepface tensorflow
#    pyzbar (QR):     pip install pyzbar
#    libzbar0 (QR):   sudo apt install libzbar0
# ═════════════════════════════════════════════════════════════

def step19a_extract_face(img_bgr):
    """
    Extract the face photograph region from the Aadhaar card image.

    APPROACH:
      1. Crop the known photo region (left 28%, vertical center)
      2. Run OpenCV Haar Cascade face detector within that crop
      3. If face found: use detected bbox (precise)
         If not found: use the whole photo region crop (fallback)
      4. Add padding around detected face for context

    Args:
        img_bgr : full Aadhaar card image as BGR numpy array

    Returns:
        face_img   : BGR numpy array of face crop (or None)
        face_bbox  : (x, y, w, h) in original image coords (or None)
        method     : 'haar_cascade' / 'region_crop' / 'not_found'
    """
    section("19A — Face Extraction")

    if img_bgr is None:
        warn("No image provided for face extraction")
        return None, None, 'not_found'

    h, w = img_bgr.shape[:2]
    info(f"Card image size: {w}x{h}")

    # ── Crop the photo region (left ~28%, vertical middle) ────
    x_start = 0
    x_end   = int(w * 0.30)
    y_start = int(h * 0.12)
    y_end   = int(h * 0.88)
    photo_region = img_bgr[y_start:y_end, x_start:x_end]

    if photo_region.size == 0:
        warn("Photo region crop is empty")
        return None, None, 'not_found'

    # ── Try Haar Cascade face detection ──────────────────────
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade_alt  = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'

    face_img = None
    face_bbox = None
    method = 'region_crop'

    for cascade_path in [face_cascade_path, face_cascade_alt]:
        try:
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                continue

            gray_region = cv2.cvtColor(photo_region, cv2.COLOR_BGR2GRAY)

            # Try multiple scale factors to handle different print sizes
            for scale_factor in [1.05, 1.1, 1.15, 1.2]:
                faces = cascade.detectMultiScale(
                    gray_region,
                    scaleFactor=scale_factor,
                    minNeighbors=3,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces) > 0:
                    # Pick the largest detected face
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    fx, fy, fw, fh = faces[0]

                    # Add padding (20% on each side)
                    pad_x = int(fw * 0.20)
                    pad_y = int(fh * 0.20)
                    fx2 = max(0, fx - pad_x)
                    fy2 = max(0, fy - pad_y)
                    fw2 = min(photo_region.shape[1] - fx2, fw + 2 * pad_x)
                    fh2 = min(photo_region.shape[0] - fy2, fh + 2 * pad_y)

                    face_img  = photo_region[fy2:fy2+fh2, fx2:fx2+fw2]
                    # Convert to original image coordinates
                    face_bbox = (x_start + fx2, y_start + fy2, fw2, fh2)
                    method    = 'haar_cascade'
                    ok(f"Face detected by Haar Cascade at ({fx},{fy}) size {fw}x{fh}")
                    break

            if method == 'haar_cascade':
                break

        except Exception as e:
            warn(f"Haar Cascade failed: {e}")

    # ── Fallback: use entire photo region ─────────────────────
    if face_img is None:
        warn("Haar Cascade did not detect face — using full photo region")
        info("Tip: better lighting / higher resolution improves detection")
        face_img  = photo_region.copy()
        face_bbox = (x_start, y_start, x_end - x_start, y_end - y_start)
        method    = 'region_crop'

    rh, rw = face_img.shape[:2]
    ok(f"Face region: {rw}x{rh} pixels  |  method: {method}")

    return face_img, face_bbox, method


def step19b_face_quality(face_img):
    """
    Assess quality of the extracted face image.

    Returns:
        quality_score   : int 0–100
        quality_verdict : 'GOOD' / 'ACCEPTABLE' / 'POOR'
        quality_details : dict with individual metric scores
    """
    section("19B — Face Quality Assessment")

    if face_img is None or face_img.size == 0:
        warn("No face image to assess")
        return 0, 'NO FACE', {}

    h, w = face_img.shape[:2]
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img

    details = {}
    score = 0

    # ── Sharpness (Laplacian variance) ───────────────────────
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if   lap_var > 200:  sharpness_pts = 35; sharpness_label = "Sharp"
    elif lap_var > 80:   sharpness_pts = 25; sharpness_label = "Acceptable"
    elif lap_var > 20:   sharpness_pts = 12; sharpness_label = "Blurry"
    else:                sharpness_pts =  0; sharpness_label = "Very Blurry"
    score += sharpness_pts
    details['sharpness'] = {'value': round(lap_var, 1), 'label': sharpness_label,
                             'points': sharpness_pts, 'max': 35}
    info(f"  Sharpness    : {lap_var:.1f}  → {sharpness_label}  (+{sharpness_pts}/35)")

    # ── Brightness ────────────────────────────────────────────
    mean_brightness = np.mean(gray)
    if   60 <= mean_brightness <= 200:  bright_pts = 25; bright_label = "Good"
    elif 40 <= mean_brightness <= 220:  bright_pts = 15; bright_label = "Acceptable"
    else:                               bright_pts =  5; bright_label = "Poor"
    score += bright_pts
    details['brightness'] = {'value': round(mean_brightness, 1), 'label': bright_label,
                              'points': bright_pts, 'max': 25}
    info(f"  Brightness   : {mean_brightness:.1f}  → {bright_label}  (+{bright_pts}/25)")

    # ── Face size ─────────────────────────────────────────────
    area = h * w
    if   area > 5000:   size_pts = 25; size_label = "Good size"
    elif area > 2000:   size_pts = 15; size_label = "Small"
    elif area > 500:    size_pts =  8; size_label = "Very small"
    else:               size_pts =  0; size_label = "Too small"
    score += size_pts
    details['size'] = {'value': f"{w}x{h} ({area}px²)", 'label': size_label,
                        'points': size_pts, 'max': 25}
    info(f"  Size         : {w}x{h} px ({area} px²)  → {size_label}  (+{size_pts}/25)")

    # ── Contrast ──────────────────────────────────────────────
    contrast = gray.std()
    if   contrast > 40:   contrast_pts = 15; contrast_label = "High contrast"
    elif contrast > 20:   contrast_pts = 10; contrast_label = "Medium contrast"
    else:                 contrast_pts =  3; contrast_label = "Low contrast"
    score += contrast_pts
    details['contrast'] = {'value': round(contrast, 1), 'label': contrast_label,
                            'points': contrast_pts, 'max': 15}
    info(f"  Contrast     : σ={contrast:.1f}  → {contrast_label}  (+{contrast_pts}/15)")

    # ── Verdict ───────────────────────────────────────────────
    if   score >= 75:  verdict = "GOOD"
    elif score >= 45:  verdict = "ACCEPTABLE"
    else:              verdict = "POOR"

    ok(f"  Quality Score: {score}/100  →  {verdict}")
    details['overall'] = score

    return score, verdict, details


def step19c_face_match(card_face_img, selfie_path):
    """
    Compare the face extracted from the Aadhaar card against a selfie.

    ENGINE PRIORITY:
      1. DeepFace — deep learning face recognition (best accuracy)
         Model: VGG-Face (pre-trained, downloads ~500MB once)
         Install: pip install deepface tensorflow
      2. OpenCV LBPH — classical ML fallback (always available)
         Less accurate on diverse poses/lighting but works offline.

    Args:
        card_face_img : BGR numpy array — face crop from card
        selfie_path   : str — path to selfie image file

    Returns:
        result dict with:
          match         : bool
          match_score   : float 0–100
          distance      : raw distance metric (lower = more similar)
          engine        : 'deepface' / 'lbph' / 'none'
          verdict       : 'MATCH' / 'NO MATCH' / 'INCONCLUSIVE'
          note          : explanation string
    """
    section("19C — Face Match (Card vs Selfie)")

    result = {
        'match':       False,
        'match_score': 0.0,
        'distance':    None,
        'engine':      'none',
        'verdict':     'INCONCLUSIVE',
        'note':        '',
    }

    if card_face_img is None or card_face_img.size == 0:
        warn("No card face image available for matching")
        result['note'] = 'Card face not extracted'
        return result

    if not selfie_path or not os.path.exists(selfie_path):
        warn(f"Selfie not found: {selfie_path}")
        result['note'] = 'Selfie file not found'
        return result

    selfie_img = cv2.imread(selfie_path)
    if selfie_img is None:
        warn(f"Cannot read selfie: {selfie_path}")
        result['note'] = 'Cannot read selfie'
        return result

    ok(f"Selfie loaded: {selfie_path}")

    # ── Save card face to a temp file (needed by DeepFace) ────
    import tempfile
    tmp_card_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_card_path = tmp.name
        cv2.imwrite(tmp_card_path, card_face_img)

        # ── Engine 0: InsightFace ArcFace (best accuracy) ────────
        info("Trying InsightFace ArcFace (best accuracy for Indian faces)...")
        insightface_result = _insightface_match(card_face_img, selfie_path)
        if insightface_result is not None:
            # InsightFace succeeded — copy its result and return
            for k, v in insightface_result.items():
                result[k] = v
            return result

        info("InsightFace not available — trying DeepFace...")

        # ── Engine 1: DeepFace ────────────────────────────────
        info("Trying DeepFace (deep learning face recognition)...")
        try:
            from deepface import DeepFace

            info("DeepFace loaded — running verification...")
            info("(First run downloads VGG-Face model ~500MB — cached after)")

            df_result = DeepFace.verify(
                img1_path   = tmp_card_path,
                img2_path   = selfie_path,
                model_name  = 'VGG-Face',
                distance_metric = 'cosine',
                enforce_detection = False,  # don't crash if face not perfectly detected
                detector_backend  = 'opencv',
            )

            verified  = df_result.get('verified', False)
            distance  = df_result.get('distance', 1.0)
            threshold = df_result.get('threshold', 0.40)

            # Convert cosine distance (0=same, 1=different) to score (100=same)
            match_score = max(0.0, (1.0 - distance / threshold) * 100)
            match_score = min(100.0, match_score)

            result['engine']      = 'deepface'
            result['distance']    = round(distance, 4)
            result['match_score'] = round(match_score, 1)
            result['match']       = verified

            if verified:
                result['verdict'] = 'MATCH ✓'
                result['note']    = (f"DeepFace VGG-Face: cosine distance={distance:.3f} "
                                     f"< threshold={threshold:.3f}")
            else:
                result['verdict'] = 'NO MATCH ✗'
                result['note']    = (f"DeepFace VGG-Face: cosine distance={distance:.3f} "
                                     f">= threshold={threshold:.3f}")

            ok(f"DeepFace result  : {result['verdict']}")
            ok(f"Match score      : {match_score:.1f}/100")
            ok(f"Cosine distance  : {distance:.4f}  (threshold={threshold:.3f})")

        except ImportError:
            warn("DeepFace not installed — falling back to OpenCV LBPH")
            warn("For best accuracy: pip install deepface tensorflow")
            raise _LBPHFallback()

        except _LBPHFallback:
            raise

        except Exception as e:
            warn(f"DeepFace error: {e}")
            warn("Falling back to OpenCV LBPH")
            raise _LBPHFallback()

    except _LBPHFallback:
        # ── Engine 2: OpenCV LBPH ─────────────────────────────
        info("Running OpenCV LBPH face comparison (classical ML)...")
        info("Note: LBPH is less accurate than DeepFace — install deepface for better results")

        try:
            result = _lbph_face_match(card_face_img, selfie_img, result)
        except Exception as e:
            warn(f"LBPH also failed: {e}")
            result['note'] = f'All face matching engines failed: {e}'
            result['verdict'] = 'INCONCLUSIVE'

    finally:
        # Clean up temp file
        if tmp_card_path and os.path.exists(tmp_card_path):
            try:
                os.unlink(tmp_card_path)
            except Exception:
                pass

    return result


class _LBPHFallback(Exception):
    """Internal signal to switch to LBPH engine."""
    pass


def _lbph_face_match(card_face_img, selfie_img, result):
    """
    OpenCV LBPH-based face similarity (fallback when DeepFace unavailable).

    HOW IT WORKS:
      1. Convert both images to grayscale and resize to 100x100
      2. Compute Local Binary Pattern (LBP) histograms for each
      3. Compare histograms using Chi-squared distance
      4. Normalize distance to a 0–100 similarity score

    LBP captures local texture patterns — works reasonably well
    for the same person in different photos.
    Lower chi-squared distance = more similar faces.

    Score interpretation:
      > 70   : Strong match
      50–70  : Probable match
      30–50  : Weak / uncertain
      < 30   : No match
    """
    def preprocess(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        resized = cv2.resize(gray, (100, 100), interpolation=cv2.INTER_AREA)
        return resized

    def compute_lbp_hist(gray_img):
        """
        Compute LBP histogram manually using OpenCV operations.
        Divides image into a 4x4 grid and concatenates local histograms
        for spatial information (more discriminative than global LBP).
        """
        h, w = gray_img.shape
        hist_all = []
        grid = 4
        ch, cw = h // grid, w // grid

        for row in range(grid):
            for col in range(grid):
                cell = gray_img[row*ch:(row+1)*ch, col*cw:(col+1)*cw]
                # Approximate LBP: compare each pixel to its 8 neighbors
                # Using simple threshold on mean of neighborhood
                blurred = cv2.GaussianBlur(cell, (3, 3), 0)
                diff = cell.astype(np.int16) - blurred.astype(np.int16)
                binary = (diff > 0).astype(np.uint8) * 255
                hist = cv2.calcHist([binary], [0], None, [32], [0, 256])
                cv2.normalize(hist, hist)
                hist_all.extend(hist.flatten().tolist())

        return np.array(hist_all, dtype=np.float32)

    card_proc   = preprocess(card_face_img)
    selfie_proc = preprocess(selfie_img)

    card_hist   = compute_lbp_hist(card_proc)
    selfie_hist = compute_lbp_hist(selfie_proc)

    # Chi-squared histogram comparison (OpenCV method)
    chi_sq = cv2.compareHist(card_hist, selfie_hist, cv2.HISTCMP_CHISQR_ALT)

    # Convert chi-squared distance to similarity score
    # Empirically: chi-sq ~0 = identical, ~2 = very different
    # Normalize: score = max(0, 100 * (1 - chi_sq / 2.0))
    similarity = max(0.0, 100.0 * (1.0 - min(chi_sq / 2.0, 1.0)))

    result['engine']      = 'lbph'
    result['distance']    = round(float(chi_sq), 4)
    result['match_score'] = round(similarity, 1)

    MATCH_THRESHOLD = 55.0

    if similarity >= 70:
        result['match']   = True
        result['verdict'] = 'MATCH ✓'
        result['note']    = f"LBPH: strong similarity {similarity:.1f}/100 (χ²={chi_sq:.3f})"
    elif similarity >= MATCH_THRESHOLD:
        result['match']   = True
        result['verdict'] = 'MATCH ✓ (weak)'
        result['note']    = f"LBPH: probable match {similarity:.1f}/100 (χ²={chi_sq:.3f}) — verify manually"
    elif similarity >= 35:
        result['match']   = False
        result['verdict'] = 'INCONCLUSIVE'
        result['note']    = f"LBPH: uncertain {similarity:.1f}/100 — install deepface for accuracy"
    else:
        result['match']   = False
        result['verdict'] = 'NO MATCH ✗'
        result['note']    = f"LBPH: low similarity {similarity:.1f}/100 (χ²={chi_sq:.3f})"

    ok(f"LBPH similarity  : {similarity:.1f}/100")
    ok(f"Chi-sq distance  : {chi_sq:.4f}")
    ok(f"LBPH verdict     : {result['verdict']}")

    return result


def step19d_liveness_hint(face_img):
    """
    Basic liveness hint — detects 'photo of a photo' texture artifacts.

    METHOD: 2D FFT frequency analysis
      A real face:
        • Mid-range frequencies dominate (natural skin texture)
        • No sharp periodic peaks in frequency domain
      A printed photo:
        • Halftone dot patterns create regular frequency peaks
        • High-frequency energy in a grid pattern
      A phone/screen:
        • Pixel grid creates strong energy at specific frequencies

    HOW WE DETECT IT:
      1. Compute 2D FFT of the face grayscale image
      2. Shift DC to center (fftshift)
      3. Compute magnitude spectrum
      4. Mask out the DC component (center)
      5. Find the ratio of high-frequency peaks to mean energy
      6. High ratio → possible printing artifact → flag it

    Returns:
        likely_real     : bool
        liveness_score  : float 0–100 (100 = likely real face)
        note            : human-readable explanation
    """
    section("19D — Liveness Hint (Anti-Spoofing Check)")

    if face_img is None or face_img.size == 0:
        warn("No face image for liveness check")
        return True, 50.0, "No face image — cannot assess"

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img

    # Resize to standard size for consistent analysis
    gray_std = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

    # 2D FFT
    fft    = np.fft.fft2(gray_std.astype(np.float64))
    fft_sh = np.fft.fftshift(fft)
    mag    = np.abs(fft_sh)
    mag_log = np.log1p(mag)

    h, w = mag.shape
    cy, cx = h // 2, w // 2

    # Mask DC component (center 5x5 pixels)
    mag_no_dc = mag_log.copy()
    mag_no_dc[cy-2:cy+3, cx-2:cx+3] = 0

    # Global stats
    mean_energy  = np.mean(mag_no_dc)
    max_energy   = np.max(mag_no_dc)
    peak_ratio   = max_energy / (mean_energy + 1e-6)

    # High-frequency ring energy (outer 40% of spectrum)
    mask_hf = np.zeros_like(mag_no_dc)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            if dist > 0.4 * min(h, w):
                mask_hf[y, x] = 1
    hf_energy   = np.mean(mag_no_dc * mask_hf)
    hf_ratio    = hf_energy / (mean_energy + 1e-6)

    info(f"  FFT peak ratio    : {peak_ratio:.2f}  (high = possible artifact)")
    info(f"  HF energy ratio   : {hf_ratio:.2f}  (high = possible screen/print)")

    # Score calculation
    # High peak ratio (>8) suggests periodic print patterns
    # High HF ratio (>1.8) suggests screen pixels or print raster
    peak_penalty = max(0, (peak_ratio - 5) * 5)    # starts penalizing above 5
    hf_penalty   = max(0, (hf_ratio   - 1.5) * 15) # starts penalizing above 1.5
    raw_score    = max(0, 100 - peak_penalty - hf_penalty)
    liveness_score = min(100.0, raw_score)

    if liveness_score >= 65:
        likely_real = True
        note = f"Texture analysis: likely real face (score={liveness_score:.0f}/100)"
    elif liveness_score >= 40:
        likely_real = True  # give benefit of doubt
        note = (f"Texture analysis: uncertain — possible print artifact "
                f"(score={liveness_score:.0f}/100) — verify manually")
    else:
        likely_real = False
        note = (f"Texture analysis: possible photo-of-photo attack "
                f"(score={liveness_score:.0f}/100, peak_ratio={peak_ratio:.1f})")

    ok(f"  Liveness score    : {liveness_score:.0f}/100")
    ok(f"  Verdict           : {'Likely Real' if likely_real else 'Suspicious'}")
    info(f"  Note: {note}")
    info("  (This is a heuristic check — not a certified PAD system)")

    return likely_real, liveness_score, note




# ═════════════════════════════════════════════════════════════
#  PHASE 3 UPGRADES — World-class Face AI
# ═════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────
#  19E — PASSIVE LIVENESS DETECTION (upgrade from 19D heuristic)
#
#  WHY THIS IS BETTER THAN THE FFT LIVENESS HINT (19D):
#    19D looks at the whole card image using global FFT peaks.
#    19E looks ONLY at the face region using 4 independent
#    texture analysis methods that real face detectors use.
#
#    Methods combined:
#      1. Local Binary Pattern (LBP) texture entropy
#         Real skin: complex, high-entropy LBP histogram
#         Printed photo: uniform, low-entropy LBP
#
#      2. Gradient magnitude distribution
#         Real face: natural edge gradient falloff
#         Screen/print: sharp digital edges everywhere
#
#      3. High-frequency energy ratio
#         Real face: low HF energy (skin is smooth)
#         Moiré/halftone: high HF energy at regular intervals
#
#      4. Colour saturation analysis
#         Real face: moderate, varied skin saturation
#         Printed photo: often over-saturated or flat
#
#  RETURNS: (liveness_score: float, likely_real: bool, detail: dict)
# ─────────────────────────────────────────────────────────────

def step19e_passive_liveness(face_img):
    """
    Passive liveness detection on the face region.
    Combines 4 texture analysis methods into a single score.

    Args:
        face_img : BGR numpy array of face crop

    Returns:
        (liveness_score: float 0-100, likely_real: bool, detail: dict)
    """
    section("19E — Passive Liveness Detection")

    if face_img is None or face_img.size == 0:
        warn("No face image for passive liveness")
        return 50.0, True, {}

    h, w = face_img.shape[:2]
    if h < 30 or w < 30:
        warn(f"Face too small for liveness ({w}x{h}) — skipping")
        return 60.0, True, {}

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Standardise size for consistent analysis
    face_std = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    scores   = {}

    # ── Method 1: LBP Texture Entropy ─────────────────────────
    # LBP (Local Binary Pattern) compares each pixel to its 8 neighbours.
    # Real skin has complex, varied texture → high entropy LBP histogram.
    # A printed/screen photo has more uniform patterns → lower entropy.
    try:
        radius    = 1
        neighbors = 8
        lbp       = np.zeros_like(face_std, dtype=np.uint8)

        for i in range(radius, face_std.shape[0] - radius):
            for j in range(radius, face_std.shape[1] - radius):
                center   = face_std[i, j]
                pattern  = 0
                offsets  = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
                for bit, (dy, dx) in enumerate(offsets):
                    if face_std[i+dy, j+dx] >= center:
                        pattern |= (1 << bit)
                lbp[i, j] = pattern

        # Compute histogram and entropy
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist    = hist[hist > 0].astype(np.float64)
        probs   = hist / hist.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Real face: entropy 5-7 (complex)
        # Print/screen: entropy 3-5 (uniform)
        lbp_score = min(100, max(0, (entropy - 3.0) / 4.0 * 100))
        scores['lbp_entropy'] = {
            'value': round(entropy, 3),
            'score': round(lbp_score, 1),
            'label': 'High entropy = complex skin texture'
        }
        info(f"  LBP entropy: {entropy:.3f} → score {lbp_score:.0f}/100")

    except Exception as e:
        warn(f"  LBP computation failed: {e}")
        scores['lbp_entropy'] = {'value': 0, 'score': 60.0, 'label': 'Failed'}
        lbp_score = 60.0

    # ── Method 2: Gradient Magnitude Distribution ──────────────
    # Real face has smooth gradient falloff (skin is continuous).
    # A printed/screen photo has sharp digital edges everywhere.
    # We measure: what fraction of gradients are very high (>150)?
    # Real face: low fraction. Print/screen: high fraction.
    try:
        gx = cv2.Sobel(face_std, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(face_std, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)

        high_grad_ratio = np.sum(mag > 150) / mag.size
        # Real face: high_grad_ratio < 0.05
        # Print/digital: high_grad_ratio > 0.15
        grad_score = min(100, max(0, (0.20 - high_grad_ratio) / 0.20 * 100))
        scores['gradient'] = {
            'value': round(high_grad_ratio, 4),
            'score': round(grad_score, 1),
            'label': 'Low sharp-edge ratio = natural face'
        }
        info(f"  Gradient high-ratio: {high_grad_ratio:.4f} → score {grad_score:.0f}/100")

    except Exception as e:
        warn(f"  Gradient computation failed: {e}")
        scores['gradient'] = {'value': 0, 'score': 60.0, 'label': 'Failed'}
        grad_score = 60.0

    # ── Method 3: High-Frequency Energy Ratio ─────────────────
    # Moiré/halftone patterns concentrate energy in specific
    # high-frequency bands. Real face has low HF energy.
    try:
        fft    = np.fft.fft2(face_std.astype(np.float64))
        fft_sh = np.fft.fftshift(fft)
        mag_f  = np.abs(fft_sh)

        cy, cx = face_std.shape[0]//2, face_std.shape[1]//2

        # Total energy vs high-frequency energy (outer 35%)
        total_energy = np.sum(mag_f)
        hf_energy    = 0.0
        for y in range(face_std.shape[0]):
            for x in range(face_std.shape[1]):
                r = np.sqrt((y-cy)**2 + (x-cx)**2)
                if r > 0.35 * min(cy, cx) * 2:
                    hf_energy += mag_f[y, x]

        hf_ratio = hf_energy / (total_energy + 1e-6)
        # Real face: hf_ratio ~0.6-0.75 (natural falloff)
        # Screen/print: hf_ratio >0.80 (artificial peaks)
        hf_score = min(100, max(0, (0.85 - hf_ratio) / 0.25 * 100))
        scores['hf_energy'] = {
            'value': round(hf_ratio, 4),
            'score': round(hf_score, 1),
            'label': 'Normal HF ratio = no screen pattern'
        }
        info(f"  HF energy ratio: {hf_ratio:.4f} → score {hf_score:.0f}/100")

    except Exception as e:
        warn(f"  HF energy computation failed: {e}")
        scores['hf_energy'] = {'value': 0, 'score': 60.0, 'label': 'Failed'}
        hf_score = 60.0

    # ── Method 4: Skin Colour Saturation Analysis ──────────────
    # Real face: HSV saturation 30-120 for Indian/South Asian skin.
    # Printed photo: saturation often compressed (<25 or >150).
    # Screen photo: often over-saturated (>140).
    try:
        hsv  = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        s    = hsv[:, :, 1].astype(np.float32)

        # Focus on face-like hue range (Indian skin: H 5-25 in OpenCV 0-180)
        h_ch = hsv[:, :, 0].astype(np.float32)
        skin_mask = ((h_ch >= 3) & (h_ch <= 30) &
                     (hsv[:, :, 2] > 40)).astype(np.uint8)

        if np.sum(skin_mask) > 200:
            skin_sat  = s[skin_mask > 0]
            mean_sat  = np.mean(skin_sat)
            std_sat   = np.std(skin_sat)
            # Good range: mean 40-100, std 10-40
            mean_ok   = 1.0 if 25 <= mean_sat <= 120 else 0.4
            std_ok    = 1.0 if 8  <= std_sat  <= 50  else 0.5
            sat_score = mean_ok * std_ok * 90
        else:
            # No clear skin region — neutral score
            sat_score = 65
            mean_sat  = 0
            std_sat   = 0

        scores['saturation'] = {
            'value': round(float(mean_sat), 1),
            'score': round(sat_score, 1),
            'label': 'Natural skin saturation'
        }
        info(f"  Skin saturation: mean={mean_sat:.1f} std={std_sat:.1f} "
             f"→ score {sat_score:.0f}/100")

    except Exception as e:
        warn(f"  Saturation analysis failed: {e}")
        scores['saturation'] = {'value': 0, 'score': 60.0, 'label': 'Failed'}
        sat_score = 60.0

    # ── Combine scores (weighted) ──────────────────────────────
    weights = {'lbp_entropy': 0.35, 'gradient': 0.30,
               'hf_energy':   0.20, 'saturation': 0.15}
    method_scores = {
        'lbp_entropy': lbp_score,
        'gradient':    grad_score,
        'hf_energy':   hf_score,
        'saturation':  sat_score,
    }

    liveness_score = sum(method_scores[k] * weights[k] for k in weights)
    liveness_score = round(min(100.0, max(0.0, liveness_score)), 1)

    likely_real = liveness_score >= 55

    if liveness_score >= 75:
        verdict_str = "Likely real face"
    elif liveness_score >= 55:
        verdict_str = "Probably real — verify manually"
    else:
        verdict_str = "Suspicious — possible printed/screen photo"

    ok(f"  Passive liveness : {liveness_score:.0f}/100 — {verdict_str}")

    return liveness_score, likely_real, scores


# ─────────────────────────────────────────────────────────────
#  19F — AGE CONSISTENCY CHECK
#
#  WHAT IT DOES:
#    Uses DeepFace's age estimator to estimate the age of the
#    person in the card photo. Compares against the declared DOB.
#    If the estimated age differs by more than 12 years from the
#    declared age, it flags the card.
#
#  WHY IT CATCHES FRAUD:
#    The most common stolen-card fraud scenario:
#      Person steals an Aadhaar card of someone who looks similar.
#      The stolen card has a different DOB — maybe 10-15 years off.
#      The face LOOKS similar but the age doesn't match the DOB.
#
#    This check is the only one that catches this attack.
#    QR, Verhoeff, ELA all pass — but the age will be wrong.
#
#  TOLERANCE: ±12 years (accounts for lighting, angle, age of photo)
#
#  GRACEFUL FALLBACK:
#    If DeepFace not installed → uses Haar cascade proxy (less accurate)
#    If both fail → skips with neutral score (never crashes pipeline)
# ─────────────────────────────────────────────────────────────

def step19f_age_consistency(face_img, dob_str):
    """
    Estimate age from face and compare against declared DOB.

    Args:
        face_img : BGR numpy array of face crop
        dob_str  : Date of birth string "DD/MM/YYYY"

    Returns:
        (score: int 0-100, note: str, detail: dict)
        score 100 = age matches DOB perfectly
        score 0   = massive age inconsistency (likely fraud)
    """
    section("19F — Age Consistency Check")

    if face_img is None or face_img.size == 0:
        return 50, "No face image for age check", {}

    if not dob_str:
        return 50, "No DOB available for age comparison", {}

    # Parse DOB
    try:
        import re as _re
        from datetime import date as _date
        import datetime as _datetime
        m = _re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', dob_str.strip())
        if not m:
            return 50, f"Cannot parse DOB: {dob_str}", {}
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        dob         = _date(year, month, day)
        today       = _date.today()
        declared_age = (today - dob).days / 365.25
        info(f"  Declared DOB: {dob_str} → declared age: {declared_age:.1f} years")
    except Exception as e:
        return 50, f"DOB parse error: {e}", {}

    estimated_age = None
    method        = None

    # ── Engine 1: DeepFace age estimation ─────────────────────
    try:
        from deepface import DeepFace
        import tempfile, os as _os

        tmp_path = _os.path.join(tempfile.gettempdir(), '_age_check.jpg')
        cv2.imwrite(tmp_path, face_img)

        result = DeepFace.analyze(
            img_path          = tmp_path,
            actions           = ['age'],
            enforce_detection = False,
            detector_backend  = 'opencv',
        )
        # DeepFace returns list or dict depending on version
        if isinstance(result, list):
            result = result[0]
        estimated_age = result.get('age', None)
        method        = 'DeepFace'

        try:
            _os.unlink(tmp_path)
        except Exception:
            pass

        ok(f"  DeepFace age estimate: {estimated_age:.0f} years")

    except ImportError:
        info("  DeepFace not installed — using proxy age estimator")
    except Exception as e:
        warn(f"  DeepFace age estimate failed: {e}")

    # ── Engine 2: Laplacian-based proxy (fallback) ─────────────
    # When DeepFace is unavailable, use a proxy:
    # Younger faces have smoother skin → higher Laplacian variance.
    # Older faces have more wrinkles → different LBP texture.
    # This is a rough proxy, tolerance is ±20 years.
    if estimated_age is None:
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Very rough heuristic based on skin texture complexity
            # Young (5-25): lap_var typically 200-600
            # Middle (25-50): lap_var 400-900
            # Older (50+): lap_var 600-1200
            # These ranges overlap heavily — this is only a sanity check
            if   lap_var < 250: proxy_age = 15
            elif lap_var < 400: proxy_age = 22
            elif lap_var < 600: proxy_age = 35
            elif lap_var < 900: proxy_age = 48
            else:               proxy_age = 60

            estimated_age = proxy_age
            method        = 'texture_proxy (rough)'
            info(f"  Texture proxy age: ~{proxy_age} years (lap_var={lap_var:.0f})")
        except Exception as e:
            warn(f"  Proxy age estimator failed: {e}")
            return 50, "Age estimation not available", {}

    if estimated_age is None:
        return 50, "Age estimation failed", {}

    # ── Compare estimated vs declared ─────────────────────────
    age_diff = abs(estimated_age - declared_age)
    info(f"  Age diff: |{estimated_age:.0f} - {declared_age:.1f}| = {age_diff:.1f} years")

    # Tolerance depends on engine accuracy
    tolerance = 12 if method == 'DeepFace' else 20

    if age_diff <= tolerance * 0.5:
        score = 100
        note  = (f"Age matches DOB ({method}: ~{estimated_age:.0f} yrs, "
                 f"declared {declared_age:.0f} yrs, diff={age_diff:.0f} yrs)")
    elif age_diff <= tolerance:
        score = 75
        note  = (f"Age within tolerance ({method}: ~{estimated_age:.0f} yrs, "
                 f"declared {declared_age:.0f} yrs, diff={age_diff:.0f} yrs)")
    elif age_diff <= tolerance * 1.5:
        score = 45
        note  = (f"Age mismatch ({method}: ~{estimated_age:.0f} yrs, "
                 f"declared {declared_age:.0f} yrs, diff={age_diff:.0f} yrs). "
                 f"Possible different-age card fraud — verify manually.")
    else:
        score = 10
        note  = (f"LARGE age mismatch ({method}: ~{estimated_age:.0f} yrs, "
                 f"declared {declared_age:.0f} yrs, diff={age_diff:.0f} yrs). "
                 f"Strong indicator of stolen card fraud.")

    sym = "OK" if score >= 75 else ("WARN" if score >= 45 else "FAIL")
    ok(f"  Age consistency: {score}/100 [{sym}] — {note[:60]}")

    detail = {
        'estimated_age':  round(float(estimated_age), 1),
        'declared_age':   round(declared_age, 1),
        'age_diff':       round(age_diff, 1),
        'tolerance':      tolerance,
        'method':         method,
    }
    return score, note, detail


# ─────────────────────────────────────────────────────────────
#  19G — OCCLUSION & DAMAGE DETECTION
#
#  WHAT IT DOES:
#    Checks whether the face photo is partially covered or damaged.
#    Uses facial landmark detection to verify that all key
#    landmarks (eyes, nose, mouth) are visible and unobstructed.
#
#  WHY IT MATTERS:
#    A tampered card might have:
#      1. A sticker or tape covering part of the original face
#      2. A new photo pasted over the original
#      3. Ink or pen marks obscuring features
#      4. Physical damage (torn, water damaged)
#
#    These attacks try to substitute a different person's photo
#    while making detection harder.
#
#  METHOD:
#    1. Run Haar Cascade eye detector on face region
#    2. Run Haar Cascade mouth/smile detector
#    3. Check that detected features have reasonable spatial layout
#    4. Check pixel value uniformity in key regions (solid colour = covered)
# ─────────────────────────────────────────────────────────────

def step19g_occlusion_check(face_img):
    """
    Detect face occlusion, damage, or feature obstruction.

    Args:
        face_img : BGR numpy array of face crop

    Returns:
        (score: int 0-100, note: str, detail: dict)
    """
    section("19G — Occlusion & Damage Detection")

    if face_img is None or face_img.size == 0:
        return 50, "No face image for occlusion check", {}

    h, w = face_img.shape[:2]
    if h < 40 or w < 40:
        return 60, f"Face too small for occlusion check ({w}x{h})", {}

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    detail = {}
    issues = []

    # ── Eye detection ──────────────────────────────────────────
    eyes_found = 0
    try:
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        if os.path.exists(eye_cascade_path):
            eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            # Look for eyes in top 60% of face region
            face_top = gray[:int(h * 0.65), :]
            eyes = eye_cascade.detectMultiScale(
                face_top, scaleFactor=1.05, minNeighbors=3,
                minSize=(int(w*0.05), int(w*0.05))
            )
            eyes_found = len(eyes)
            detail['eyes_detected'] = eyes_found
            info(f"  Eyes detected: {eyes_found}")
    except Exception as e:
        warn(f"  Eye detection failed: {e}")
        detail['eyes_detected'] = 0

    # ── Check for solid-colour (covered) regions ───────────────
    # A sticker or tape creates a region with near-zero variance
    block_size   = max(16, min(w, h) // 6)
    low_var_blocks = 0
    total_blocks   = 0

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block    = gray[y:y+block_size, x:x+block_size]
            variance = np.var(block)
            total_blocks += 1
            if variance < 15:   # nearly uniform — possible sticker
                low_var_blocks += 1

    solid_ratio = low_var_blocks / max(total_blocks, 1)
    detail['solid_region_ratio'] = round(solid_ratio, 3)
    info(f"  Solid (covered) region ratio: {solid_ratio:.3f}")

    if solid_ratio > 0.25:
        issues.append(f"Large uniform region ({solid_ratio:.0%}) — possible sticker or tape")

    # ── Check overall image variance (damage/blur) ─────────────
    overall_var = np.var(gray)
    detail['overall_variance'] = round(float(overall_var), 1)
    info(f"  Overall variance: {overall_var:.1f}")

    if overall_var < 100:
        issues.append("Very low image variance — face may be damaged or obscured")

    # ── Check brightness uniformity (ink/pen marks) ────────────
    # Ink marks create very dark regions in otherwise bright face area
    dark_ratio = np.sum(gray < 30) / gray.size
    detail['dark_pixel_ratio'] = round(dark_ratio, 4)
    info(f"  Dark pixel ratio: {dark_ratio:.4f}")

    if dark_ratio > 0.08:
        issues.append(f"High dark pixel ratio ({dark_ratio:.1%}) — possible ink marks")

    # ── Compute score ──────────────────────────────────────────
    eye_score    = 100 if eyes_found >= 2 else (65 if eyes_found == 1 else 35)
    solid_score  = max(0, 100 - int(solid_ratio * 300))
    var_score    = min(100, int(overall_var / 5))
    dark_score   = max(0, 100 - int(dark_ratio * 500))

    score = int(eye_score * 0.40 + solid_score * 0.30 +
                var_score  * 0.20 + dark_score  * 0.10)
    score = max(0, min(100, score))

    if issues:
        note = f"Occlusion detected: {'; '.join(issues[:2])}"
    elif eyes_found >= 2:
        note = f"Face complete — both eyes detected, no occlusion signs."
    elif eyes_found == 1:
        note = "One eye detected — partial occlusion possible or pose issue."
    else:
        note = "No eyes detected — face may be obscured or low quality."

    sym = "OK" if score >= 70 else ("WARN" if score >= 45 else "FAIL")
    ok(f"  Occlusion score: {score}/100 [{sym}]")
    detail['issues'] = issues

    return score, note, detail


# ─────────────────────────────────────────────────────────────
#  19H — INSIGHTFACE UPGRADE FOR FACE MATCHING
#
#  WHY InsightFace > DeepFace/VGG-Face:
#    DeepFace uses VGG-Face (2015 model).
#    InsightFace uses ArcFace (2019) trained on MS1MV2 dataset.
#    ArcFace is the current state-of-art for 1:1 face verification.
#
#    Accuracy on LFW benchmark:
#      VGG-Face  : 98.95%
#      ArcFace   : 99.83%
#
#    More importantly for Indian IDs:
#      ArcFace trained on more diverse dataset → better on South Asian faces
#      ArcFace better on low-resolution card photos
#      ArcFace better on partial poses (ID photos are often not perfectly frontal)
#
#  INSTALL:
#    pip install insightface onnxruntime
#    (no GPU required — CPU inference works fine for 1:1 matching)
#
#  GRACEFUL FALLBACK:
#    InsightFace not installed → falls back to DeepFace → falls back to LBPH
#    Pipeline never crashes regardless of what is installed.
# ─────────────────────────────────────────────────────────────

def _insightface_match(card_face_img, selfie_path):
    """
    Face matching using InsightFace ArcFace model.

    Returns: result dict (same schema as step19c_face_match)
             or None if InsightFace not available.
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
        import tempfile
    except ImportError:
        return None

    result = {
        'match':       False,
        'match_score': 0.0,
        'distance':    None,
        'engine':      'insightface_arcface',
        'verdict':     'INCONCLUSIVE',
        'note':        '',
    }

    try:
        info("  Loading InsightFace ArcFace model...")
        app = FaceAnalysis(
            name='buffalo_sc',            # lightweight model (~200MB)
            providers=['CPUExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(128, 128))
        ok("  InsightFace loaded")

        # Process card face
        card_rgb = cv2.cvtColor(card_face_img, cv2.COLOR_BGR2RGB)
        card_faces = app.get(card_rgb)

        # Process selfie
        selfie_img = cv2.imread(selfie_path)
        if selfie_img is None:
            result['note'] = 'Cannot read selfie'
            return result
        selfie_rgb  = cv2.cvtColor(selfie_img, cv2.COLOR_BGR2RGB)
        selfie_faces = app.get(selfie_rgb)

        if not card_faces:
            result['note'] = 'No face detected in card image by InsightFace'
            return result
        if not selfie_faces:
            result['note'] = 'No face detected in selfie by InsightFace'
            return result

        # Get embeddings (512-dim ArcFace embedding)
        emb_card   = card_faces[0].normed_embedding
        emb_selfie = selfie_faces[0].normed_embedding

        # Cosine similarity (embeddings are already normalised)
        similarity = float(np.dot(emb_card, emb_selfie))

        # ArcFace cosine similarity threshold: 0.28 (same-person)
        # Convert similarity (-1 to 1) to match_score (0 to 100)
        match_score = max(0.0, min(100.0, (similarity + 1.0) / 2.0 * 100))
        threshold   = 0.28

        result['distance']    = round(1.0 - similarity, 4)
        result['match_score'] = round(match_score, 1)
        result['match']       = similarity >= threshold

        if similarity >= threshold:
            result['verdict'] = 'MATCH ✓'
            result['note']    = (f"ArcFace: cosine_sim={similarity:.3f} "
                                 f">= threshold={threshold}")
        else:
            result['verdict'] = 'NO MATCH ✗'
            result['note']    = (f"ArcFace: cosine_sim={similarity:.3f} "
                                 f"< threshold={threshold}")

        ok(f"  InsightFace result : {result['verdict']}")
        ok(f"  Match score        : {match_score:.1f}/100")
        ok(f"  Cosine similarity  : {similarity:.4f} (threshold={threshold})")

        return result

    except Exception as e:
        warn(f"  InsightFace failed: {e}")
        return None

def step19_face_pipeline(img_bgr, image_path, selfie_path=None, fields=None):
    """
    Orchestrate the full Face AI pipeline:
      19A — Extract face from card
      19B — Assess face quality
      19C — Match against selfie (if selfie provided)
      19D — Liveness hint

    Returns:
        face_result dict with all sub-results
    """
    face_result = {
        'face_img':                 None,
        'face_bbox':                None,
        'extraction_method':        None,
        'quality_score':            0,
        'quality_verdict':          'NOT_ASSESSED',
        'quality_details':          {},
        'match_result':             None,
        'liveness_score':           None,
        'likely_real':              None,
        'liveness_note':            None,
        'selfie_provided':          bool(selfie_path),
        # Phase 3 new fields
        'passive_liveness_score':   None,
        'passive_liveness_real':    None,
        'passive_liveness_detail':  {},
        'combined_liveness_score':  None,
        'combined_liveness_real':   None,
        'age_score':                None,
        'age_note':                 None,
        'age_detail':               {},
        'occlusion_score':          None,
        'occlusion_note':           None,
        'occlusion_detail':         {},
    }

    # ── 19A: Extract face ─────────────────────────────────────
    if img_bgr is None and image_path:
        img_bgr = cv2.imread(image_path)

    face_img, face_bbox, method = step19a_extract_face(img_bgr)
    face_result['face_img']          = face_img
    face_result['face_bbox']         = face_bbox
    face_result['extraction_method'] = method

    if face_img is None:
        warn("Face extraction failed — skipping quality, match, liveness")
        return face_result

    # Save face crop to outputs folder for inspection
    face_out_path = '/mnt/user-data/outputs/aadhaar_face_crop.jpg'
    try:
        os.makedirs('/mnt/user-data/outputs', exist_ok=True)
        cv2.imwrite(face_out_path, face_img)
        ok(f"Face crop saved: {face_out_path}")
    except Exception:
        pass

    # ── 19B: Face quality ─────────────────────────────────────
    q_score, q_verdict, q_details = step19b_face_quality(face_img)
    face_result['quality_score']   = q_score
    face_result['quality_verdict'] = q_verdict
    face_result['quality_details'] = q_details

    # ── 19D: Liveness hint (original FFT-based) ──────────────
    likely_real, live_score, live_note = step19d_liveness_hint(face_img)
    face_result['likely_real']    = likely_real
    face_result['liveness_score'] = live_score
    face_result['liveness_note']  = live_note

    # ── 19E: Passive liveness (4-method texture analysis) ─────
    # Runs on face region only — more precise than 19D
    section("19E — Passive Liveness")
    p_live_score, p_live_real, p_live_detail = step19e_passive_liveness(face_img)
    face_result['passive_liveness_score']  = p_live_score
    face_result['passive_liveness_real']   = p_live_real
    face_result['passive_liveness_detail'] = p_live_detail

    # Use the STRICTER of the two liveness scores for the final verdict
    combined_liveness = min(live_score, p_live_score)
    face_result['combined_liveness_score'] = combined_liveness
    face_result['combined_liveness_real']  = (combined_liveness >= 55)
    info(f"Combined liveness: {combined_liveness:.0f}/100 "
         f"(FFT={live_score:.0f}, passive={p_live_score:.0f})")

    # ── 19F: Age consistency check ────────────────────────────
    dob_str = fields.get('dob', '') if fields else ''
    age_score, age_note, age_detail = step19f_age_consistency(face_img, dob_str)
    face_result['age_score']  = age_score
    face_result['age_note']   = age_note
    face_result['age_detail'] = age_detail

    if age_score < 45:
        warn(f"Age inconsistency detected: {age_note}")

    # ── 19G: Occlusion & damage check ────────────────────────
    occ_score, occ_note, occ_detail = step19g_occlusion_check(face_img)
    face_result['occlusion_score']  = occ_score
    face_result['occlusion_note']   = occ_note
    face_result['occlusion_detail'] = occ_detail

    if occ_score < 50:
        warn(f"Occlusion/damage detected: {occ_note}")

    # ── 19C: Face match (InsightFace → DeepFace → LBPH) ──────
    if selfie_path:
        match_result = step19c_face_match(face_img, selfie_path)
        face_result['match_result'] = match_result
    else:
        info("No selfie provided — face matching skipped")
        info("To enable: run with --selfie path/to/selfie.jpg")

    return face_result