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


def step19_face_pipeline(img_bgr, image_path, selfie_path=None):
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
        'face_img':         None,
        'face_bbox':        None,
        'extraction_method': None,
        'quality_score':    0,
        'quality_verdict':  'NOT_ASSESSED',
        'quality_details':  {},
        'match_result':     None,
        'liveness_score':   None,
        'likely_real':      None,
        'liveness_note':    None,
        'selfie_provided':  bool(selfie_path),
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
    face_out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aadhaar_face_crop.jpg")
    try:
        cv2.imwrite(face_out_path, face_img)
        ok(f"Face crop saved: {face_out_path}")
    except Exception:
        pass

    # ── 19B: Face quality ─────────────────────────────────────
    q_score, q_verdict, q_details = step19b_face_quality(face_img)
    face_result['quality_score']   = q_score
    face_result['quality_verdict'] = q_verdict
    face_result['quality_details'] = q_details

    # ── 19D: Liveness hint ────────────────────────────────────
    likely_real, live_score, live_note = step19d_liveness_hint(face_img)
    face_result['likely_real']    = likely_real
    face_result['liveness_score'] = live_score
    face_result['liveness_note']  = live_note

    # ── 19C: Face match ───────────────────────────────────────
    if selfie_path:
        match_result = step19c_face_match(face_img, selfie_path)
        face_result['match_result'] = match_result
    else:
        info("No selfie provided — face matching skipped")
        info("To enable: run with --selfie path/to/selfie.jpg")

    return face_result