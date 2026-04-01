# qr_verification.py
# Step 18: QR Code Extraction & Cross-Verification
#   Helpers: _deblur_wiener(), _detect_qr_opencv(), _detect_qr_pyzbar(), _parse_aadhaar_qr()
#   Comparators: _normalize_name_for_compare(), _names_match(),
#                _dobs_match(), _aadhaar_nums_match()
#   Main: step18_qr_verify()
# ─────────────────────────────────────────────────────────────
# Updated: Maximum QR detection — 30+ preprocessing variants
#          including region crop, inversion, gamma, bilateral,
#          rotation, and WeChatQRCode detector.

import cv2
import re
import numpy as np

from utils import section, ok, info, warn, err

# ═════════════════════════════════════════════════════════════
#  STEP 18 — QR Code Extraction & Cross-Verification
#
#  WHY THIS IS YOUR MOAT:
#    Every genuine Aadhaar card has a QR code printed on it.
#    The QR encodes the cardholder's data (name, DOB, gender,
#    address) in a structured XML/string format, optionally
#    signed by UIDAI. A tampered document (edited photo,
#    photoshopped number, AI-generated fake) will either:
#      (a) have no valid QR at all, or
#      (b) have a QR whose contents don't match the printed text
#
#  TRUST SCORE FORMULA:
#    ┌─────────────────────────────────────────────────────┐
#    │  QR Found           : +20 pts                       │
#    │  QR Decoded         : +10 pts                       │
#    │  Aadhaar# matches   : +35 pts  (highest weight)     │
#    │  Name matches       : +20 pts                       │
#    │  DOB matches        : +10 pts                       │
#    │  Gender matches     : +5  pts                       │
#    │                     ─────                           │
#    │  Max possible       : 100 pts                       │
#    └─────────────────────────────────────────────────────┘
#    Score ≥ 80 → LIKELY GENUINE
#    Score 50–79 → REVIEW REQUIRED
#    Score < 50 → FRAUD SUSPECTED
#
#  INSTALL (optional, improves QR detection on blurry images):
#    pip install pyzbar pillow numpy
#    Ubuntu: sudo apt install libzbar0
# ═════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────
#  BLUR / EXPOSURE PREPROCESSING HELPERS
# ─────────────────────────────────────────────────────────────

def _deblur_wiener(gray, kernel_size=5, noise_power=0.01):
    """
    Frequency-domain Wiener filter to reverse Gaussian / camera blur.
    """
    try:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel /= kernel.sum()
        img_f    = np.fft.fft2(gray.astype(np.float32) / 255.0)
        ker_f    = np.fft.fft2(kernel, s=gray.shape)
        ker_conj = np.conj(ker_f)
        wiener   = ker_conj / (np.abs(ker_f) ** 2 + noise_power)
        result   = np.real(np.fft.ifft2(img_f * wiener))
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    except Exception:
        return gray


def _unsharp_mask(gray, sigma=3, strength=1.8):
    """Unsharp masking to enhance edges."""
    blurred   = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharpened = cv2.addWeighted(gray, strength, blurred, -(strength - 1), 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _gamma_correct(gray, gamma):
    """
    Gamma correction — brightens dark images (gamma < 1)
    or darkens overexposed images (gamma > 1).

    Formula: output = (input / 255) ^ (1/gamma) * 255

    Why this helps QR detection:
      A dark scan makes QR finder patterns blend into background.
      Gamma < 1 brightens shadows, revealing the white cells.
      Gamma > 1 darkens a washed-out (overexposed) image,
      increasing contrast between black/white QR modules.
    """
    inv_gamma = 1.0 / gamma
    table     = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(gray, table)


def _bilateral_denoise(gray):
    """
    Bilateral filter — removes noise while PRESERVING hard edges.

    Unlike Gaussian blur (which blurs everything), bilateral filter
    only averages pixels that are both spatially close AND similar
    in intensity. This keeps QR finder-pattern edges sharp while
    smoothing out JPEG compression artifacts and paper texture.

    Why this helps QR detection:
      JPEG compression creates 8×8 block artifacts that confuse
      QR decoders into seeing false patterns. Bilateral removes
      these without blurring the actual QR module boundaries.
    """
    return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)


def _crop_qr_region(img_bgr):
    """
    Crop the region most likely to contain the Aadhaar QR code.

    Aadhaar card layout (back side):
      - The large QR code is always in the BOTTOM-RIGHT quadrant
      - Roughly: right 45% of width, bottom 55% of height
      - There's also a smaller QR on the front side (bottom-right)

    Why crop helps:
      The full card image has address text, logos, barcodes, and
      the UIDAI hologram — all of which add noise for the QR
      detector. Cropping to just the QR region dramatically
      reduces false positives and speeds up detection.

    Returns:
      List of (label, cropped_bgr) tuples to try.
    """
    h, w = img_bgr.shape[:2]
    crops = []

    # Back side — large QR: bottom-right
    crops.append(("qr_region_br",
                  img_bgr[int(h*0.45):h, int(w*0.55):w]))

    # Back side — slightly wider crop (in case card is slightly rotated)
    crops.append(("qr_region_br_wide",
                  img_bgr[int(h*0.35):h, int(w*0.45):w]))

    # Front side — small QR: bottom-right corner
    crops.append(("qr_region_front_br",
                  img_bgr[int(h*0.65):h, int(w*0.65):w]))

    # Full right half (covers both orientations)
    crops.append(("right_half",
                  img_bgr[:, int(w*0.5):w]))

    # Full bottom half
    crops.append(("bottom_half",
                  img_bgr[int(h*0.5):h, :]))

    # Filter out empty crops
    return [(lbl, c) for lbl, c in crops
            if c is not None and c.size > 0 and c.shape[0] > 20 and c.shape[1] > 20]


def _build_preprocessing_variants(img_bgr):
    """
    Generate ALL preprocessing variants to try.
    Ordered from cheapest/most-likely to most expensive/last-resort.

    Each variant is (label, bgr_image).
    The caller tries each in order and stops at the first QR decode.

    TOTAL VARIANTS: ~50 across all combinations.
    """
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variants = []

    # ── Tier 1: Direct (fast) ────────────────────────────────
    variants.append(("raw",         img_bgr))
    variants.append(("raw_gray",    cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))

    # ── Tier 2: Inversion ────────────────────────────────────
    # Some Aadhaar printouts have white QR on dark background,
    # or a scan inverts the image. Always try inverted.
    inv_gray = cv2.bitwise_not(gray)
    variants.append(("inverted",    cv2.cvtColor(inv_gray, cv2.COLOR_GRAY2BGR)))

    # ── Tier 3: Gamma correction ─────────────────────────────
    # Dark scan: gamma=0.4 brightens; overexposed: gamma=2.0 darkens
    for gamma in [0.4, 0.6, 1.5, 2.0]:
        g = _gamma_correct(gray, gamma)
        variants.append((f"gamma_{gamma}",
                         cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)))

    # ── Tier 4: CLAHE ────────────────────────────────────────
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl_gray = clahe.apply(gray)
    variants.append(("clahe",       cv2.cvtColor(cl_gray, cv2.COLOR_GRAY2BGR)))

    # Tighter CLAHE (8px tiles — better for small QR modules)
    clahe2   = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    cl2_gray = clahe2.apply(gray)
    variants.append(("clahe_tight", cv2.cvtColor(cl2_gray, cv2.COLOR_GRAY2BGR)))

    # ── Tier 5: Bilateral denoise ────────────────────────────
    bil_gray = _bilateral_denoise(gray)
    variants.append(("bilateral",   cv2.cvtColor(bil_gray, cv2.COLOR_GRAY2BGR)))

    # Bilateral then CLAHE — JPEG artifact removal then contrast
    bil_cl   = clahe.apply(bil_gray)
    variants.append(("bilateral+clahe",
                     cv2.cvtColor(bil_cl, cv2.COLOR_GRAY2BGR)))

    # ── Tier 6: Unsharp mask ─────────────────────────────────
    usm_gray = _unsharp_mask(gray, sigma=3, strength=1.8)
    variants.append(("unsharp_mask",
                     cv2.cvtColor(usm_gray, cv2.COLOR_GRAY2BGR)))

    # Stronger sharpening for very blurry images
    usm_strong = _unsharp_mask(gray, sigma=5, strength=2.5)
    variants.append(("unsharp_strong",
                     cv2.cvtColor(usm_strong, cv2.COLOR_GRAY2BGR)))

    # CLAHE + unsharp mask
    cl_usm   = _unsharp_mask(cl_gray, sigma=3, strength=1.8)
    variants.append(("clahe+usm",   cv2.cvtColor(cl_usm, cv2.COLOR_GRAY2BGR)))

    # ── Tier 7: Wiener deconvolution ─────────────────────────
    wiener_gray   = _deblur_wiener(gray, kernel_size=5, noise_power=0.01)
    wiener_strong = _deblur_wiener(gray, kernel_size=9, noise_power=0.005)
    variants.append(("wiener",        cv2.cvtColor(wiener_gray,   cv2.COLOR_GRAY2BGR)))
    variants.append(("wiener_strong", cv2.cvtColor(wiener_strong, cv2.COLOR_GRAY2BGR)))

    # ── Tier 8: Binarization ─────────────────────────────────
    _, otsu = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu",         cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)))
    variants.append(("otsu_inv",     cv2.cvtColor(cv2.bitwise_not(otsu),
                                                   cv2.COLOR_GRAY2BGR)))

    # Adaptive threshold — handles uneven illumination
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    variants.append(("adaptive_thresh",
                     cv2.cvtColor(adapt, cv2.COLOR_GRAY2BGR)))

    # Adaptive on CLAHE image
    adapt_cl = cv2.adaptiveThreshold(
        cl_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    variants.append(("clahe+adaptive",
                     cv2.cvtColor(adapt_cl, cv2.COLOR_GRAY2BGR)))

    # Adaptive on bilateral-denoised image
    adapt_bil = cv2.adaptiveThreshold(
        bil_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    variants.append(("bilateral+adaptive",
                     cv2.cvtColor(adapt_bil, cv2.COLOR_GRAY2BGR)))

    # ── Tier 9: Morphological cleanup ────────────────────────
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph_open  = cv2.morphologyEx(otsu, cv2.MORPH_OPEN,  k3)
    morph_close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k2)
    variants.append(("morph_open",
                     cv2.cvtColor(morph_open,  cv2.COLOR_GRAY2BGR)))
    variants.append(("morph_close",
                     cv2.cvtColor(morph_close, cv2.COLOR_GRAY2BGR)))

    # ── Tier 10: QR region crops + preprocessing ─────────────
    # Try every crop region with the top preprocessing variants
    for crop_label, crop_img in _crop_qr_region(img_bgr):
        if crop_img.size == 0:
            continue
        crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        variants.append((f"{crop_label}_raw",  crop_img))
        variants.append((f"{crop_label}_inv",
                         cv2.cvtColor(cv2.bitwise_not(crop_gray),
                                      cv2.COLOR_GRAY2BGR)))

        _, c_otsu = cv2.threshold(crop_gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((f"{crop_label}_otsu",
                         cv2.cvtColor(c_otsu, cv2.COLOR_GRAY2BGR)))

        c_cl = clahe.apply(crop_gray)
        variants.append((f"{crop_label}_clahe",
                         cv2.cvtColor(c_cl, cv2.COLOR_GRAY2BGR)))

        c_usm = _unsharp_mask(crop_gray, sigma=3, strength=1.8)
        variants.append((f"{crop_label}_usm",
                         cv2.cvtColor(c_usm, cv2.COLOR_GRAY2BGR)))

    # ── Tier 11: Rotation variants ───────────────────────────
    # If the card was photographed sideways or upside-down,
    # the QR decoder will fail even on a clear image.
    for angle, label in [(90,  "rot90"),
                         (180, "rot180"),
                         (270, "rot270")]:
        rotated = cv2.rotate(img_bgr, {
            90:  cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }[angle])
        variants.append((label, rotated))

        # Also try Otsu on each rotation
        rot_gray  = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        _, rot_otsu = cv2.threshold(rot_gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((f"{label}_otsu",
                         cv2.cvtColor(rot_otsu, cv2.COLOR_GRAY2BGR)))

    # ── Tier 12: Scale cascade ───────────────────────────────
    # Most effective on low-resolution images (< 400 px QR region).
    # Upscaling gives the QR decoder more pixels to work with.
    h, w = img_bgr.shape[:2]
    for scale in [1.5, 2.0, 3.0]:
        up      = cv2.resize(img_bgr,
                             (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
        up_gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        up_usm  = _unsharp_mask(up_gray, sigma=3, strength=1.8)
        variants.append((f"upscale_{scale}x+usm",
                         cv2.cvtColor(up_usm, cv2.COLOR_GRAY2BGR)))

        _, up_otsu = cv2.threshold(up_usm, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((f"upscale_{scale}x+otsu",
                         cv2.cvtColor(up_otsu, cv2.COLOR_GRAY2BGR)))

        # Also try crops on the upscaled image
        for crop_label, crop_img in _crop_qr_region(up):
            if crop_img.size == 0:
                continue
            crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            _, c_otsu = cv2.threshold(crop_gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append((f"upscale_{scale}x+{crop_label}_otsu",
                             cv2.cvtColor(c_otsu, cv2.COLOR_GRAY2BGR)))

    return variants


# ─────────────────────────────────────────────────────────────
#  QR DETECTION ENGINES
# ─────────────────────────────────────────────────────────────

def _try_wechat_qr(img_bgr):
    """
    Try WeChatQRCode detector from opencv-contrib-python.
    FAR more robust than the built-in QRCodeDetector —
    uses a deep learning model trained on real-world QR codes.

    Install: pip install opencv-contrib-python
    (replaces opencv-python — uninstall it first)

    Returns (text, bbox) or (None, None).
    """
    try:
        detector = cv2.wechat_qrcode_WeChatQRCode()
        texts, points = detector.detectAndDecode(img_bgr)
        if texts:
            return texts[0], points[0] if points else None
    except (AttributeError, cv2.error):
        pass
    return None, None


def _detect_qr_opencv(img_bgr):
    """
    Attempt QR detection using OpenCV's built-in QRCodeDetector,
    trying every preprocessing variant in order.
    Also attempts WeChatQRCode if opencv-contrib is installed.

    Returns (raw_str, bbox, variant_label) or (None, None, None).
    """
    # Try WeChatQRCode first on raw image — if available it's the best
    text, bbox = _try_wechat_qr(img_bgr)
    if text:
        return text, bbox, "WeChatQRCode_raw"

    detector = cv2.QRCodeDetector()

    for label, variant_img in _build_preprocessing_variants(img_bgr):
        # Try WeChatQRCode on each variant too
        text, bbox = _try_wechat_qr(variant_img)
        if text:
            return text, bbox, f"WeChatQRCode_{label}"

        # Built-in OpenCV detector
        try:
            data, bbox, _ = detector.detectAndDecode(variant_img)
            if data and bbox is not None:
                return data, bbox, label
        except Exception:
            pass

    return None, None, None


def _detect_qr_pyzbar(img_bgr):
    """
    Attempt QR detection using pyzbar, trying every preprocessing variant.
    Requires: pip install pyzbar   +   sudo apt install libzbar0

    Returns list of (raw_data_str, polygon, variant_label) or empty list.
    """
    try:
        from pyzbar import pyzbar
        from PIL import Image as PILImage

        for label, variant_img in _build_preprocessing_variants(img_bgr):
            pil_img = PILImage.fromarray(
                cv2.cvtColor(variant_img, cv2.COLOR_BGR2RGB)
            )
            decoded = pyzbar.decode(pil_img)
            results = [
                (obj.data.decode('utf-8', errors='replace'), obj.polygon, label)
                for obj in decoded if obj.type == 'QRCODE'
            ]
            if results:
                return results

        return []
    except ImportError:
        return []
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────
#  QR PAYLOAD PARSER
# ─────────────────────────────────────────────────────────────

def _parse_aadhaar_qr(raw_data):
    """
    Parse the raw QR payload into a structured dict.

    Aadhaar QR formats seen in the wild:
      Format A — XML (older cards):
        <PrintLetterBarcodeData uid="XXXX XXXX XXXX"
          name="FULL NAME" dob="DD/MM/YYYY" gender="M/F/T" ... />

      Format B — Pipe-delimited string (newer secure QR):
        <version>|<uid_last4>|<name>|<dob>|<gender>|<co>|<house>|
        <street>|<lm>|<loc>|<vtc>|<subdist>|<dist>|<state>|<pc>|
        <mobile_last4>

      Format C — JSON (rare, some states):
        {"name": "...", "dob": "...", ...}

    Returns dict with keys: name, dob, gender, aadhaar_last4,
    aadhaar_number (if found), address_parts, mobile_last4
    """
    if not raw_data:
        return {}

    parsed = {}
    raw    = raw_data.strip()

    # ── Format A: XML ────────────────────────────────────────
    if raw.startswith('<') or 'PrintLetterBarcodeData' in raw:
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(raw)
            attr = root.attrib

            uid = attr.get('uid', attr.get('Uid', ''))
            if uid:
                parsed['aadhaar_number'] = uid.replace('-', ' ')
                digits = re.sub(r'\D', '', uid)
                if len(digits) >= 4:
                    parsed['aadhaar_last4'] = digits[-4:]

            name = attr.get('name', attr.get('Name', ''))
            if name:
                parsed['name'] = name.strip().upper()

            dob = attr.get('dob', attr.get('Dob', attr.get('DOB', '')))
            if dob:
                parsed['dob'] = dob.strip()

            gender_raw = attr.get('gender', attr.get('Gender', ''))
            if gender_raw:
                g = gender_raw.strip().upper()
                parsed['gender'] = (
                    'Male'        if g in ('M', 'MALE')        else
                    'Female'      if g in ('F', 'FEMALE')      else
                    'Transgender' if g in ('T', 'TRANSGENDER') else g
                )

            addr_parts = []
            for key in ('co', 'house', 'street', 'lm', 'loc', 'vtc',
                        'subdist', 'dist', 'state', 'pc'):
                v = attr.get(key, attr.get(key.capitalize(), ''))
                if v and v.strip():
                    addr_parts.append(v.strip())
            if addr_parts:
                parsed['address_parts'] = addr_parts

            mobile = attr.get('mobile', attr.get('Mobile', ''))
            if mobile and len(re.sub(r'\D', '', mobile)) >= 4:
                parsed['mobile_last4'] = re.sub(r'\D', '', mobile)[-4:]

        except Exception as e:
            warn(f"    QR XML parse error: {e}")

    # ── Format B: Pipe-delimited ─────────────────────────────
    elif '|' in raw and not raw.startswith('{'):
        parts = raw.split('|')
        if len(parts) >= 5:
            try:
                last4_or_uid = parts[1].strip()
                if re.match(r'^\d{4,12}$', last4_or_uid):
                    parsed['aadhaar_last4'] = last4_or_uid[-4:]

                name = parts[2].strip()
                if name and re.match(r'^[A-Za-z\s.\-]+$', name):
                    parsed['name'] = name.upper()

                dob = parts[3].strip()
                if re.match(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{4}', dob):
                    parsed['dob'] = dob

                gender_raw = parts[4].strip().upper()
                parsed['gender'] = (
                    'Male'        if gender_raw in ('M', 'MALE', '1')        else
                    'Female'      if gender_raw in ('F', 'FEMALE', '2')      else
                    'Transgender' if gender_raw in ('T', 'TRANSGENDER', '3') else None
                )

                addr_parts = [p.strip() for p in parts[5:] if p.strip()]
                if addr_parts:
                    parsed['address_parts'] = addr_parts

                if addr_parts:
                    last_part = addr_parts[-1]
                    if re.match(r'^\d{4}$', last_part):
                        parsed['mobile_last4']  = last_part
                        parsed['address_parts'] = addr_parts[:-1]

            except Exception as e:
                warn(f"    QR pipe-delimited parse error: {e}")

    # ── Format C: JSON ───────────────────────────────────────
    elif raw.startswith('{'):
        try:
            import json
            data = json.loads(raw)

            name = data.get('name', data.get('Name', ''))
            if name:
                parsed['name'] = str(name).strip().upper()

            dob = data.get('dob', data.get('DOB', data.get('dateOfBirth', '')))
            if dob:
                parsed['dob'] = str(dob).strip()

            gender_raw = data.get('gender', data.get('Gender', ''))
            if gender_raw:
                g = str(gender_raw).strip().upper()
                parsed['gender'] = (
                    'Male'        if g in ('M', 'MALE')        else
                    'Female'      if g in ('F', 'FEMALE')      else
                    'Transgender' if g in ('T', 'TRANSGENDER') else g
                )

            uid = data.get('uid', data.get('UID', data.get('aadhaarNumber', '')))
            if uid:
                parsed['aadhaar_number'] = str(uid).replace('-', ' ')
                digits = re.sub(r'\D', '', str(uid))
                if len(digits) >= 4:
                    parsed['aadhaar_last4'] = digits[-4:]

        except Exception as e:
            warn(f"    QR JSON parse error: {e}")

    # ── Fallback: regex mining ───────────────────────────────
    if not parsed:
        m = re.search(r'\b([2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4})\b', raw)
        if m:
            parsed['aadhaar_number'] = m.group(1).replace('-', ' ')

        m = re.search(r'\b([A-Z]{2,}\s[A-Z]{2,}(?:\s[A-Z]{2,})?)\b', raw)
        if m:
            parsed['name'] = m.group(1)

        m = re.search(r'\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b', raw)
        if m:
            parsed['dob'] = m.group(1)

    return parsed


# ─────────────────────────────────────────────────────────────
#  FIELD COMPARATORS
# ─────────────────────────────────────────────────────────────

def _normalize_name_for_compare(name):
    if not name:
        return ''
    n = re.sub(r"[^A-Za-z\s]", "", name.upper())
    return re.sub(r'\s+', ' ', n).strip()


def _names_match(ocr_name, qr_name, threshold=0.75):
    a = _normalize_name_for_compare(ocr_name)
    b = _normalize_name_for_compare(qr_name)

    if not a or not b:
        return False, 0.0, "one side empty"
    if a == b:
        return True, 1.0, "exact match"

    def bigrams(s):
        s = s.replace(' ', '')
        return set(s[i:i+2] for i in range(len(s) - 1))

    bg_a, bg_b = bigrams(a), bigrams(b)
    if not bg_a or not bg_b:
        return a in b or b in a, 0.5, "substring check"

    intersection = len(bg_a & bg_b)
    union        = len(bg_a | bg_b)
    sim          = intersection / union if union > 0 else 0.0

    if sim >= threshold:
        return True, sim, f"fuzzy match ({sim:.0%})"
    return False, sim, f"mismatch ({sim:.0%})"


def _dobs_match(ocr_dob, qr_dob):
    if not ocr_dob or not qr_dob:
        return False, "one side empty"

    a = re.sub(r'[\-/\.]', '/', ocr_dob.strip())
    b = re.sub(r'[\-/\.]', '/', qr_dob.strip())

    if a == b:
        return True, "exact match"

    parts_a = a.split('/')
    parts_b = b.split('/')
    if len(parts_a) == 3 and len(parts_b) == 3:
        if parts_a[2] == parts_b[2]:
            if parts_a[0] == parts_b[1] and parts_a[1] == parts_b[0]:
                return True, "match (day/month order differs)"

    return False, f"mismatch: OCR='{a}'  QR='{b}'"


def _aadhaar_nums_match(ocr_num, qr_data):
    if not ocr_num:
        return False, "OCR number not found"

    ocr_digits = re.sub(r'\D', '', ocr_num)

    qr_full = qr_data.get('aadhaar_number', '')
    if qr_full:
        qr_digits = re.sub(r'\D', '', qr_full)
        if len(qr_digits) == 12 and len(ocr_digits) == 12:
            if qr_digits == ocr_digits:
                return True, "full 12-digit match ✓"
            diffs = sum(a != b for a, b in zip(qr_digits, ocr_digits))
            if diffs == 1:
                return True, "match with 1 OCR digit error (likely misread)"
            return False, f"mismatch: OCR={ocr_digits}  QR={qr_digits}"

    qr_last4 = qr_data.get('aadhaar_last4', '')
    if qr_last4 and len(ocr_digits) == 12:
        if ocr_digits[-4:] == qr_last4:
            return True, f"last-4 match ({qr_last4}) ✓  [QR is masked]"
        return False, f"last-4 mismatch: OCR ends {ocr_digits[-4:]}  QR={qr_last4}"

    return False, "QR has no Aadhaar number field"


# ─────────────────────────────────────────────────────────────
#  MAIN STEP
# ─────────────────────────────────────────────────────────────

def step18_qr_verify(image_path_or_array, ocr_fields):
    """
    Step 18 — QR Code Extraction & Cross-Verification.

    Args:
        image_path_or_array : path string or BGR numpy array
        ocr_fields          : dict from step14_extract / step14b_correct

    Returns:
        qr_result dict with keys:
          qr_found         : bool
          qr_decoded       : bool
          qr_raw           : raw QR string (or None)
          qr_fields        : parsed QR fields dict
          qr_detect_source : which engine + variant found the QR
          field_checks     : {field_name: {match, note}}
          trust_score      : int 0–100
          verdict          : str
          fraud_signals    : list of str
    """
    section("18 — QR Code Cross-Verification")
    info("Scanning image for QR code (50+ preprocessing variants)...")

    # ── Load image ────────────────────────────────────────────
    if isinstance(image_path_or_array, str):
        img_bgr = cv2.imread(image_path_or_array)
        if img_bgr is None:
            warn("Cannot load image for QR scan")
    else:
        img_bgr = image_path_or_array

    qr_result = {
        'qr_found':          False,
        'qr_decoded':        False,
        'qr_raw':            None,
        'qr_fields':         {},
        'qr_detect_source':  None,
        'field_checks':      {},
        'trust_score':       0,
        'verdict':           'UNDETERMINED',
        'fraud_signals':     [],
    }

    if img_bgr is None:
        warn("No image available for QR scan — skipping")
        qr_result['verdict'] = 'UNDETERMINED (no image)'
        return qr_result

    # ── Try QR detection ──────────────────────────────────────
    raw_qr = None
    source = None

    # Engine 1: pyzbar (most robust — tries all 50+ variants)
    pyzbar_results = _detect_qr_pyzbar(img_bgr)
    if pyzbar_results:
        raw_qr, _, variant_label = pyzbar_results[0]
        source = f"pyzbar [{variant_label}]"
        ok(f"QR detected by pyzbar  ({len(raw_qr)} bytes)  variant={variant_label}")

    # Engine 2: OpenCV + WeChatQRCode (also tries all variants)
    if not raw_qr:
        raw_qr, _, variant_label = _detect_qr_opencv(img_bgr)
        if raw_qr:
            source = f"OpenCV [{variant_label}]"
            ok(f"QR detected by OpenCV  ({len(raw_qr)} bytes)  variant={variant_label}")

    if not raw_qr:
        warn("No QR code detected after 50+ preprocessing variants")
        warn("Possible reasons:")
        warn("  • Back side of card not provided (QR is on the back)")
        warn("  • Image resolution too low  (QR needs ≥ 150 DPI)")
        warn("  • QR code region is physically damaged or covered")
        warn("  • Digital screenshot / photocopy without QR embedded")
        warn("  TIP: Install pyzbar for better detection:")
        warn("       pip install pyzbar")
        warn("       (Windows) pip install python-zxing")
        warn("  TIP: Install opencv-contrib for WeChatQRCode:")
        warn("       pip uninstall opencv-python")
        warn("       pip install opencv-contrib-python")
        qr_result['fraud_signals'].append(
            "QR code not detected — cannot verify document authenticity"
        )
        qr_result['trust_score'] = 35
        qr_result['verdict']     = 'REVIEW REQUIRED (QR not found)'
        return qr_result

    qr_result['qr_found']         = True
    qr_result['qr_raw']           = raw_qr
    qr_result['qr_detect_source'] = source
    info(f"QR source  : {source}")
    info(f"QR snippet : {raw_qr[:120]}...")

    # ── Parse QR payload ─────────────────────────────────────
    info("Parsing QR payload...")
    qr_fields = _parse_aadhaar_qr(raw_qr)
    qr_result['qr_fields'] = qr_fields

    if not qr_fields:
        warn("QR found but payload could not be parsed")
        qr_result['fraud_signals'].append(
            "QR code present but unreadable — structure may be altered"
        )
        qr_result['trust_score'] = 40
        qr_result['verdict']     = 'REVIEW REQUIRED (QR unreadable)'
        return qr_result

    qr_result['qr_decoded'] = True
    ok(f"QR decoded successfully — {len(qr_fields)} fields extracted")
    for k, v in qr_fields.items():
        if k != 'address_parts':
            info(f"  QR.{k:<20}: {v}")
    if 'address_parts' in qr_fields:
        info(f"  QR.address         : {', '.join(qr_fields['address_parts'][:4])}...")

    # ── Cross-check fields ────────────────────────────────────
    print()
    info("Cross-checking QR fields against OCR fields...")
    print()

    trust_score   = 20 + 10   # QR found (+20) + decoded (+10)
    field_checks  = {}
    fraud_signals = []

    # ── Aadhaar Number ────────────────────────────────────────
    aadhaar_match, aadhaar_note = _aadhaar_nums_match(
        ocr_fields.get('aadhaar_number'), qr_fields
    )
    field_checks['aadhaar_number'] = {'match': aadhaar_match, 'note': aadhaar_note}
    if aadhaar_match:
        trust_score += 35
        ok(f"  Aadhaar#  [MATCH  ] : {aadhaar_note}")
    else:
        print(f"  [XX]  Aadhaar#  [MISMATCH] : {aadhaar_note}")
        fraud_signals.append(f"Aadhaar number mismatch — {aadhaar_note}")

    # ── Name ──────────────────────────────────────────────────
    if qr_fields.get('name') and ocr_fields.get('name'):
        name_match, name_sim, name_note = _names_match(
            ocr_fields['name'], qr_fields['name']
        )
        field_checks['name'] = {'match': name_match, 'note': name_note}
        if name_match:
            trust_score += 20
            ok(f"  Name      [MATCH  ] : OCR='{ocr_fields['name']}'  "
               f"QR='{qr_fields['name']}'  ({name_note})")
        else:
            print(f"  [XX]  Name      [MISMATCH] : OCR='{ocr_fields['name']}'  "
                  f"QR='{qr_fields['name']}'  ({name_note})")
            fraud_signals.append(
                f"Name mismatch — OCR: '{ocr_fields['name']}', "
                f"QR: '{qr_fields['name']}' ({name_note})"
            )
    else:
        field_checks['name'] = {'match': None, 'note': 'not available in QR'}
        info("  Name      [SKIP   ] : QR name field not present")

    # ── DOB ───────────────────────────────────────────────────
    if qr_fields.get('dob') and ocr_fields.get('dob'):
        dob_match, dob_note = _dobs_match(ocr_fields['dob'], qr_fields['dob'])
        field_checks['dob'] = {'match': dob_match, 'note': dob_note}
        if dob_match:
            trust_score += 10
            ok(f"  DOB       [MATCH  ] : {dob_note}  "
               f"OCR='{ocr_fields['dob']}'  QR='{qr_fields['dob']}'")
        else:
            print(f"  [XX]  DOB       [MISMATCH] : {dob_note}")
            fraud_signals.append(f"DOB mismatch — {dob_note}")
    else:
        field_checks['dob'] = {'match': None, 'note': 'not available in QR'}
        info("  DOB       [SKIP   ] : QR DOB field not present")

    # ── Gender ────────────────────────────────────────────────
    if qr_fields.get('gender') and ocr_fields.get('gender'):
        qr_gen  = qr_fields['gender'].capitalize()
        ocr_gen = ocr_fields['gender'].capitalize()
        gen_match = (qr_gen == ocr_gen)
        field_checks['gender'] = {
            'match': gen_match,
            'note':  'match' if gen_match else f"OCR='{ocr_gen}'  QR='{qr_gen}'"
        }
        if gen_match:
            trust_score += 5
            ok(f"  Gender    [MATCH  ] : '{qr_gen}'")
        else:
            print(f"  [XX]  Gender    [MISMATCH] : OCR='{ocr_gen}'  QR='{qr_gen}'")
            fraud_signals.append(
                f"Gender mismatch — OCR: '{ocr_gen}', QR: '{qr_gen}'"
            )
    else:
        field_checks['gender'] = {'match': None, 'note': 'not available in QR'}
        info("  Gender    [SKIP   ] : QR gender field not present")

    # ── Clamp & verdict ───────────────────────────────────────
    trust_score = min(trust_score, 100)
    qr_result['trust_score']   = trust_score
    qr_result['field_checks']  = field_checks
    qr_result['fraud_signals'] = fraud_signals

    if trust_score >= 80:
        verdict = "LIKELY GENUINE ✓"
    elif trust_score >= 50:
        verdict = "REVIEW REQUIRED ⚠"
    else:
        verdict = "FRAUD SUSPECTED ✗"

    qr_result['verdict'] = verdict

    print()
    print(f"  {'─'*54}")
    print(f"  Trust Score    : {trust_score}/100")
    print(f"  Verdict        : {verdict}")
    print(f"  Detect source  : {source}")
    if fraud_signals:
        print(f"  Fraud Signals ({len(fraud_signals)}):")
        for sig in fraud_signals:
            print(f"    ⚠  {sig}")
    print(f"  {'─'*54}")

    return qr_result