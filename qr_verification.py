# qr_verification.py
# Step 18: QR Code Extraction & Cross-Verification
#
# WHAT'S NEW — Secure QR V5 support:
#   Real Aadhaar cards since ~2019 use "Secure QR" format.
#   The QR contains a large decimal integer which, when decoded,
#   is a gzip/zlib compressed pipe-delimited payload with ALL
#   cardholder fields — no encryption, no UIDAI SDK needed.
#
#   Fields extracted from Secure QR V5:
#     name, dob, gender, address (full structured), PIN,
#     district, state, mobile_last4, reference_id
#
#   All fields are now cross-checked against OCR output.
#
# TRUST SCORE (updated):
#   QR found           : +10 pts
#   QR decoded         : +10 pts
#   Name matches       : +20 pts
#   DOB matches        : +20 pts
#   Gender matches     : +10 pts
#   PIN matches        : +10 pts
#   District matches   : +10 pts
#   State matches      : +10 pts
#   Max possible       : 100 pts
#
#   Score >= 80 -> LIKELY GENUINE
#   Score 50-79 -> REVIEW REQUIRED
#   Score < 50  -> FRAUD SUSPECTED
#
# INSTALL:
#   pip install pyzbar pillow opencv-python
#   Ubuntu: sudo apt install libzbar0
# ─────────────────────────────────────────────────────────────

import cv2
import re
import zlib
import numpy as np

from utils import section, ok, info, warn, err


# ─────────────────────────────────────────────────────────────
#  SECURE QR DECODER  (the core new feature)
# ─────────────────────────────────────────────────────────────

def _decode_secure_qr(raw_bytes_or_str):
    """
    Decode UIDAI Secure QR (V2-V5) payload into a field dict.

    HOW IT WORKS:
      The QR stores a large decimal integer as an ASCII string.
      Converting that integer to bytes gives a gzip/zlib-compressed
      pipe-delimited (0xFF separator) record:

        V5 | 2 | <ref_id> | <name> | <dob> | <gender> | (co) |
        <district> | <landmark> | <building> | <locality> |
        <pin> | <vtc> | <state> | <street> | <subdist> |
        <dist2> | <mobile_masked> | (email_flag)

      Fields are separated by 0xFF bytes.
      No encryption — plain zlib compression only.

    Args:
        raw_bytes_or_str : bytes or str from pyzbar / OpenCV

    Returns:
        dict with all extracted fields, or {} on failure
    """
    if not raw_bytes_or_str:
        return {}

    # Get the digit string
    if isinstance(raw_bytes_or_str, bytes):
        try:
            digit_str = raw_bytes_or_str.decode('ascii').strip()
        except Exception:
            return {}
    else:
        digit_str = str(raw_bytes_or_str).strip()

    if not digit_str.isdigit():
        return {}

    info(f"  Secure QR detected: {len(digit_str)} digit integer")

    # Convert big integer to bytes
    try:
        big_int = int(digit_str)
        n_bytes = (big_int.bit_length() + 7) // 8
        raw     = big_int.to_bytes(n_bytes, byteorder='big')
    except Exception as e:
        warn(f"  Secure QR int->bytes failed: {e}")
        return {}

    # Decompress — try all zlib modes and byte offsets
    decompressed = None
    for skip in range(0, min(15, len(raw))):
        for wbits in [47, 15, -15]:
            try:
                dec = zlib.decompress(raw[skip:], wbits)
                if len(dec) > 10:
                    decompressed = dec
                    info(f"  Decompressed at offset={skip} wbits={wbits}: {len(dec)} bytes")
                    break
            except Exception:
                pass
        if decompressed:
            break

    if not decompressed:
        warn("  Secure QR decompression failed")
        return {}

    # Split on 0xFF byte delimiter
    parts = decompressed.split(b'\xff')
    fields_raw = []
    for p in parts:
        try:
            fields_raw.append(p.decode('utf-8', errors='replace').strip())
        except Exception:
            fields_raw.append('')

    info(f"  Parsed {len(fields_raw)} raw fields from Secure QR")
    for i, f in enumerate(fields_raw[:20]):
        if f and not any(ord(c) < 9 for c in f):
            info(f"    [{i:02d}] {f[:80]}")

    def _get(idx, default=''):
        if idx < len(fields_raw):
            v = fields_raw[idx].strip()
            # Skip binary garbage
            if any(ord(c) < 32 and c not in '\t\n\r' for c in v):
                return default
            return v
        return default

    # V5 field layout (0-indexed after 0xFF split):
    #  0  version e.g. "V5"
    #  1  format marker e.g. "2"
    #  2  reference_id
    #  3  name
    #  4  dob  (DD-MM-YYYY)
    #  5  gender (M/F/T)
    #  6  care_of (C/O)
    #  7  district
    #  8  landmark
    #  9  building / house name
    #  10 locality / village
    #  11 PIN code
    #  12 vtc / sub-district
    #  13 state
    #  14 full street address
    #  15 sub-district 2
    #  16 district 2
    #  17 mobile (masked e.g. XXXXXX4051)
    #  18 (empty)
    #  19 email flag (O=not registered, Y=registered)

    version = _get(0)

    # Normalise gender
    g_raw   = _get(5).upper()
    gender_map = {
        'M': 'Male', 'MALE': 'Male',
        'F': 'Female', 'FEMALE': 'Female',
        'T': 'Transgender', 'TRANSGENDER': 'Transgender',
    }
    gender = gender_map.get(g_raw, g_raw.capitalize() if g_raw else '')

    # Normalise DOB: QR uses DD-MM-YYYY -> DD/MM/YYYY
    dob_raw = _get(4)
    dob     = re.sub(r'[-.]', '/', dob_raw) if dob_raw else ''

    # Mobile last 4
    mobile_raw   = _get(17)
    mobile_last4 = ''
    if mobile_raw:
        digits = re.sub(r'\D', '', mobile_raw)
        mobile_last4 = digits[-4:] if len(digits) >= 4 else digits

    email_flag       = _get(19)
    email_registered = (email_flag.upper() == 'Y')

    result = {
        'qr_version':       version,
        'reference_id':     _get(2),
        'name':             _get(3).upper(),
        'dob':              dob,
        'gender':           gender,
        'care_of':          _get(6),
        'district':         _get(7),
        'landmark':         _get(8),
        'building':         _get(9),
        'locality':         _get(10),
        'pin':              _get(11),
        'vtc':              _get(12),
        'state':            _get(13),
        'street':           _get(14),
        'subdistrict':      _get(15),
        'district2':        _get(16),
        'mobile_masked':    mobile_raw,
        'mobile_last4':     mobile_last4,
        'email_registered': email_registered,
        # Aadhaar number intentionally absent from Secure QR (UIDAI privacy)
        'aadhaar_number':   None,
        'aadhaar_last4':    None,
    }

    ok(f"  Secure QR decoded: name='{result['name']}' dob='{result['dob']}' "
       f"gender='{result['gender']}' pin='{result['pin']}'")

    return result


# ─────────────────────────────────────────────────────────────
#  PREPROCESSING HELPERS
# ─────────────────────────────────────────────────────────────

def _deblur_wiener(gray, kernel_size=5, noise_power=0.01):
    try:
        kernel   = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel  /= kernel.sum()
        img_f    = np.fft.fft2(gray.astype(np.float32) / 255.0)
        ker_f    = np.fft.fft2(kernel, s=gray.shape)
        ker_conj = np.conj(ker_f)
        wiener   = ker_conj / (np.abs(ker_f) ** 2 + noise_power)
        result   = np.real(np.fft.ifft2(img_f * wiener))
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    except Exception:
        return gray


def _unsharp_mask(gray, sigma=3, strength=1.8):
    blurred   = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharpened = cv2.addWeighted(gray, strength, blurred, -(strength - 1), 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _gamma_correct(gray, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(gray, table)


def _bilateral_denoise(gray):
    return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)


# ─────────────────────────────────────────────────────────────
#  QR REGION CROPS
# ─────────────────────────────────────────────────────────────

def _crop_qr_region(img_bgr):
    h, w = img_bgr.shape[:2]
    crops = [
        ("qr_region_br",       img_bgr[int(h*0.45):h,  int(w*0.55):w]),
        ("qr_region_br_wide",  img_bgr[int(h*0.35):h,  int(w*0.45):w]),
        ("qr_region_front_br", img_bgr[int(h*0.65):h,  int(w*0.65):w]),
        ("right_half",         img_bgr[:,               int(w*0.5):w]),
        ("bottom_half",        img_bgr[int(h*0.5):h,   :]),
        ("full",               img_bgr),
    ]
    return [(lbl, c) for lbl, c in crops
            if c is not None and c.size > 0
            and c.shape[0] > 20 and c.shape[1] > 20]


# ─────────────────────────────────────────────────────────────
#  PREPROCESSING VARIANTS
#  Upscale tiers come FIRST because Secure QR is very dense
#  and needs at least 2x resolution to decode reliably.
# ─────────────────────────────────────────────────────────────

def _build_preprocessing_variants(img_bgr):
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w   = img_bgr.shape[:2]
    clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    variants = []

    # Tier 0: Upscaled — most important for Secure QR
    for scale in [2, 3, 4]:
        up      = cv2.resize(img_bgr, (w * scale, h * scale),
                             interpolation=cv2.INTER_CUBIC)
        up_gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

        _, up_otsu = cv2.threshold(up_gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((f"up{scale}x_otsu",
                         cv2.cvtColor(up_otsu, cv2.COLOR_GRAY2BGR)))

        up_usm = _unsharp_mask(up_gray, sigma=2, strength=1.5)
        _, up_usm_otsu = cv2.threshold(up_usm, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((f"up{scale}x_usm_otsu",
                         cv2.cvtColor(up_usm_otsu, cv2.COLOR_GRAY2BGR)))

        up_cl = clahe.apply(up_gray)
        variants.append((f"up{scale}x_clahe",
                         cv2.cvtColor(up_cl, cv2.COLOR_GRAY2BGR)))

        up_adap = cv2.adaptiveThreshold(
            up_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)
        variants.append((f"up{scale}x_adaptive",
                         cv2.cvtColor(up_adap, cv2.COLOR_GRAY2BGR)))

        for crop_label, crop_img in _crop_qr_region(up):
            cg = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            _, co = cv2.threshold(cg, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append((f"up{scale}x_{crop_label}_otsu",
                             cv2.cvtColor(co, cv2.COLOR_GRAY2BGR)))

    # Tier 1: Raw
    variants.append(("raw",      img_bgr))
    variants.append(("raw_gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))

    # Tier 2: Inversion
    inv_gray = cv2.bitwise_not(gray)
    variants.append(("inverted", cv2.cvtColor(inv_gray, cv2.COLOR_GRAY2BGR)))

    # Tier 3: Gamma
    for gamma in [0.4, 0.6, 1.5, 2.0]:
        g = _gamma_correct(gray, gamma)
        variants.append((f"gamma_{gamma}",
                         cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)))

    # Tier 4: CLAHE
    cl_gray  = clahe.apply(gray)
    cl2_gray = clahe2.apply(gray)
    variants.append(("clahe",       cv2.cvtColor(cl_gray,  cv2.COLOR_GRAY2BGR)))
    variants.append(("clahe_tight", cv2.cvtColor(cl2_gray, cv2.COLOR_GRAY2BGR)))

    # Tier 5: Bilateral
    bil_gray = _bilateral_denoise(gray)
    bil_cl   = clahe.apply(bil_gray)
    variants.append(("bilateral",
                     cv2.cvtColor(bil_gray, cv2.COLOR_GRAY2BGR)))
    variants.append(("bilateral+clahe",
                     cv2.cvtColor(bil_cl,   cv2.COLOR_GRAY2BGR)))

    # Tier 6: Unsharp mask
    usm_gray   = _unsharp_mask(gray, sigma=3, strength=1.8)
    usm_strong = _unsharp_mask(gray, sigma=5, strength=2.5)
    cl_usm     = _unsharp_mask(cl_gray, sigma=3, strength=1.8)
    variants.append(("unsharp_mask",
                     cv2.cvtColor(usm_gray,   cv2.COLOR_GRAY2BGR)))
    variants.append(("unsharp_strong",
                     cv2.cvtColor(usm_strong, cv2.COLOR_GRAY2BGR)))
    variants.append(("clahe+usm",
                     cv2.cvtColor(cl_usm,     cv2.COLOR_GRAY2BGR)))

    # Tier 7: Wiener
    wiener_gray   = _deblur_wiener(gray, kernel_size=5, noise_power=0.01)
    wiener_strong = _deblur_wiener(gray, kernel_size=9, noise_power=0.005)
    variants.append(("wiener",
                     cv2.cvtColor(wiener_gray,   cv2.COLOR_GRAY2BGR)))
    variants.append(("wiener_strong",
                     cv2.cvtColor(wiener_strong, cv2.COLOR_GRAY2BGR)))

    # Tier 8: Binarization
    _, otsu = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt    = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    adapt_cl = cv2.adaptiveThreshold(
        cl_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    adapt_bil = cv2.adaptiveThreshold(
        bil_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    variants.append(("otsu",
                     cv2.cvtColor(otsu,      cv2.COLOR_GRAY2BGR)))
    variants.append(("otsu_inv",
                     cv2.cvtColor(cv2.bitwise_not(otsu), cv2.COLOR_GRAY2BGR)))
    variants.append(("adaptive",
                     cv2.cvtColor(adapt,     cv2.COLOR_GRAY2BGR)))
    variants.append(("clahe+adaptive",
                     cv2.cvtColor(adapt_cl,  cv2.COLOR_GRAY2BGR)))
    variants.append(("bilateral+adaptive",
                     cv2.cvtColor(adapt_bil, cv2.COLOR_GRAY2BGR)))

    # Tier 9: Morphological
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    variants.append(("morph_open",
                     cv2.cvtColor(
                         cv2.morphologyEx(otsu, cv2.MORPH_OPEN,  k3),
                         cv2.COLOR_GRAY2BGR)))
    variants.append(("morph_close",
                     cv2.cvtColor(
                         cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k2),
                         cv2.COLOR_GRAY2BGR)))

    # Tier 10: QR region crops on original
    for crop_label, crop_img in _crop_qr_region(img_bgr):
        if crop_img.size == 0:
            continue
        cg = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        variants.append((f"{crop_label}_raw",  crop_img))
        variants.append((f"{crop_label}_inv",
                         cv2.cvtColor(cv2.bitwise_not(cg),
                                      cv2.COLOR_GRAY2BGR)))
        _, co = cv2.threshold(cg, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((f"{crop_label}_otsu",
                         cv2.cvtColor(co, cv2.COLOR_GRAY2BGR)))
        cc = clahe.apply(cg)
        variants.append((f"{crop_label}_clahe",
                         cv2.cvtColor(cc, cv2.COLOR_GRAY2BGR)))

    # Tier 11: Rotation
    for angle, label, rot_code in [
        (90,  "rot90",  cv2.ROTATE_90_CLOCKWISE),
        (180, "rot180", cv2.ROTATE_180),
        (270, "rot270", cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]:
        rotated  = cv2.rotate(img_bgr, rot_code)
        rot_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        _, rot_otsu = cv2.threshold(rot_gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((label, rotated))
        variants.append((f"{label}_otsu",
                         cv2.cvtColor(rot_otsu, cv2.COLOR_GRAY2BGR)))

    return variants


# ─────────────────────────────────────────────────────────────
#  QR DETECTION ENGINES
# ─────────────────────────────────────────────────────────────

def _try_wechat_qr(img_bgr):
    try:
        detector = cv2.wechat_qrcode_WeChatQRCode()
        texts, points = detector.detectAndDecode(img_bgr)
        if texts:
            return texts[0], points[0] if points else None
    except (AttributeError, cv2.error):
        pass
    return None, None


def _detect_qr_opencv(img_bgr):
    text, bbox = _try_wechat_qr(img_bgr)
    if text:
        return text, bbox, "WeChatQRCode_raw"
    detector = cv2.QRCodeDetector()
    for label, variant_img in _build_preprocessing_variants(img_bgr):
        text, bbox = _try_wechat_qr(variant_img)
        if text:
            return text, bbox, f"WeChatQRCode_{label}"
        try:
            data, bbox, _ = detector.detectAndDecode(variant_img)
            if data and bbox is not None:
                return data, bbox, label
        except Exception:
            pass
    return None, None, None


def _detect_qr_pyzbar(img_bgr):
    try:
        from pyzbar import pyzbar
        from PIL import Image as PILImage
        for label, variant_img in _build_preprocessing_variants(img_bgr):
            pil_img = PILImage.fromarray(
                cv2.cvtColor(variant_img, cv2.COLOR_BGR2RGB))
            decoded = pyzbar.decode(pil_img)
            results = [
                (obj.data, obj.polygon, label)
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
#  LEGACY QR PARSER  (XML / pipe-text / JSON — older cards)
# ─────────────────────────────────────────────────────────────

def _parse_legacy_qr(raw_data):
    if not raw_data:
        return {}
    raw = (raw_data.decode('utf-8', errors='replace')
           if isinstance(raw_data, bytes) else str(raw_data)).strip()
    parsed = {}

    # Format A: XML
    if raw.startswith('<') or 'PrintLetterBarcodeData' in raw:
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(raw)
            attr = root.attrib
            uid = attr.get('uid', attr.get('Uid', ''))
            if uid:
                parsed['aadhaar_number'] = uid.replace('-', ' ')
                digits = re.sub(r'\D', '', uid)
                parsed['aadhaar_last4'] = digits[-4:] if len(digits) >= 4 else ''
            name = attr.get('name', attr.get('Name', ''))
            if name:
                parsed['name'] = name.strip().upper()
            dob = attr.get('dob', attr.get('DOB', ''))
            if dob:
                parsed['dob'] = re.sub(r'[-.]', '/', dob.strip())
            g = attr.get('gender', attr.get('Gender', '')).upper()
            parsed['gender'] = ('Male'        if g in ('M', 'MALE')        else
                                'Female'      if g in ('F', 'FEMALE')      else
                                'Transgender' if g in ('T', 'TRANSGENDER') else g)
            parsed['district'] = attr.get('dist', '')
            parsed['state']    = attr.get('state', '')
            parsed['pin']      = attr.get('pc', '')
            parsed['locality'] = attr.get('vtc', attr.get('loc', ''))
            parsed['landmark'] = attr.get('lm', '')
            parsed['building'] = attr.get('house', '')
            parsed['street']   = attr.get('street', '')
            parsed['care_of']  = attr.get('co', '')
            mobile = attr.get('mobile', '')
            if mobile:
                digits = re.sub(r'\D', '', mobile)
                parsed['mobile_last4'] = digits[-4:] if len(digits) >= 4 else ''
        except Exception as e:
            warn(f"  Legacy XML parse error: {e}")

    # Format B: Pipe plain text
    elif '|' in raw and not raw.startswith('{'):
        parts = raw.split('|')
        if len(parts) >= 5:
            try:
                last4 = parts[1].strip()
                if re.match(r'^\d{4,12}$', last4):
                    parsed['aadhaar_last4'] = last4[-4:]
                name = parts[2].strip()
                if name and re.match(r'^[A-Za-z\s.\-]+$', name):
                    parsed['name'] = name.upper()
                dob = parts[3].strip()
                if re.match(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{4}', dob):
                    parsed['dob'] = re.sub(r'[-.]', '/', dob)
                g = parts[4].strip().upper()
                parsed['gender'] = ('Male'        if g in ('M','MALE','1')        else
                                    'Female'      if g in ('F','FEMALE','2')      else
                                    'Transgender' if g in ('T','TRANSGENDER','3') else None)
                addr = [p.strip() for p in parts[5:] if p.strip()]
                parsed['pin']   = next((p for p in addr if re.match(r'^\d{6}$', p)), '')
                parsed['state'] = addr[-2] if len(addr) >= 2 else ''
            except Exception as e:
                warn(f"  Legacy pipe parse error: {e}")

    # Format C: JSON
    elif raw.startswith('{'):
        try:
            import json
            data = json.loads(raw)
            name = data.get('name', data.get('Name', ''))
            if name:
                parsed['name'] = str(name).strip().upper()
            dob = data.get('dob', data.get('DOB', ''))
            if dob:
                parsed['dob'] = re.sub(r'[-.]', '/', str(dob).strip())
            g = data.get('gender', data.get('Gender', '')).upper()
            parsed['gender'] = ('Male' if g in ('M','MALE') else
                                'Female' if g in ('F','FEMALE') else g)
            uid = data.get('uid', data.get('UID', ''))
            if uid:
                parsed['aadhaar_number'] = str(uid).replace('-', ' ')
                digits = re.sub(r'\D', '', str(uid))
                parsed['aadhaar_last4'] = digits[-4:] if len(digits) >= 4 else ''
            parsed['pin']   = str(data.get('pc', data.get('pin', '')))
            parsed['state'] = str(data.get('state', data.get('State', '')))
        except Exception as e:
            warn(f"  Legacy JSON parse error: {e}")

    # Fallback: regex mining
    if not parsed:
        m = re.search(r'\b([2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4})\b', raw)
        if m:
            parsed['aadhaar_number'] = m.group(1).replace('-', ' ')
        m = re.search(r'\b([A-Z]{2,}\s[A-Z]{2,}(?:\s[A-Z]{2,})?)\b', raw)
        if m:
            parsed['name'] = m.group(1)
        m = re.search(r'\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b', raw)
        if m:
            parsed['dob'] = re.sub(r'[-.]', '/', m.group(1))

    return parsed


# ─────────────────────────────────────────────────────────────
#  UNIFIED QR PARSER
# ─────────────────────────────────────────────────────────────

def _parse_aadhaar_qr(raw_data):
    """
    Try Secure QR decode first (numeric digit string >= 100 digits).
    Fall back to legacy XML/pipe/JSON parser for older cards.
    """
    if not raw_data:
        return {}

    digit_str = (raw_data.decode('ascii', errors='ignore')
                 if isinstance(raw_data, bytes) else str(raw_data)).strip()

    if digit_str.isdigit() and len(digit_str) >= 100:
        result = _decode_secure_qr(raw_data)
        if result:
            result['_format'] = 'secure_qr'
            return result

    result = _parse_legacy_qr(raw_data)
    if result:
        result['_format'] = 'legacy'
    return result


# ─────────────────────────────────────────────────────────────
#  FIELD COMPARATORS
# ─────────────────────────────────────────────────────────────

def _normalize_name(name):
    if not name:
        return ''
    n = re.sub(r"[^A-Za-z\s]", "", name.upper())
    return re.sub(r'\s+', ' ', n).strip()


def _names_match(ocr_name, qr_name, threshold=0.75):
    a = _normalize_name(ocr_name)
    b = _normalize_name(qr_name)
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
    pa, pb = a.split('/'), b.split('/')
    if len(pa) == 3 and len(pb) == 3:
        if pa[2] == pb[2] and pa[0] == pb[1] and pa[1] == pb[0]:
            return True, "match (day/month swapped)"
    return False, f"mismatch: OCR='{a}' QR='{b}'"


def _pins_match(ocr_pin, qr_pin):
    if not ocr_pin or not qr_pin:
        return False, "one side empty"
    a = re.sub(r'\D', '', str(ocr_pin).strip())
    b = re.sub(r'\D', '', str(qr_pin).strip())
    if a == b and len(a) == 6:
        return True, f"exact match ({a})"
    return False, f"mismatch: OCR='{a}' QR='{b}'"


def _location_match(ocr_val, qr_val):
    if not ocr_val or not qr_val:
        return False, "one side empty"
    a = ocr_val.strip().lower()
    b = qr_val.strip().lower()
    if a == b:
        return True, "exact match"
    if a in b or b in a:
        return True, f"partial match ('{ocr_val}' ~ '{qr_val}')"
    return False, f"mismatch: OCR='{ocr_val}' QR='{qr_val}'"


def _aadhaar_nums_match(ocr_num, qr_fields):
    if not ocr_num:
        return False, "OCR number not found"
    ocr_digits = re.sub(r'\D', '', ocr_num)

    qr_full = qr_fields.get('aadhaar_number', '')
    if qr_full:
        qr_digits = re.sub(r'\D', '', qr_full)
        if len(qr_digits) == 12 and len(ocr_digits) == 12:
            if qr_digits == ocr_digits:
                return True, "full 12-digit match"
            diffs = sum(a != b for a, b in zip(qr_digits, ocr_digits))
            if diffs == 1:
                return True, "match with 1 OCR digit error"
            return False, f"mismatch: OCR={ocr_digits} QR={qr_digits}"

    qr_last4 = qr_fields.get('aadhaar_last4', '')
    if qr_last4 and len(ocr_digits) == 12:
        if ocr_digits[-4:] == qr_last4:
            return True, f"last-4 match ({qr_last4}) [QR is masked]"
        return False, f"last-4 mismatch: OCR={ocr_digits[-4:]} QR={qr_last4}"

    # Secure QR intentionally omits Aadhaar number
    if qr_fields.get('_format') == 'secure_qr':
        return None, "Aadhaar# not stored in Secure QR (UIDAI privacy design)"

    return False, "QR has no Aadhaar number field"


# ─────────────────────────────────────────────────────────────
#  MAIN STEP
# ─────────────────────────────────────────────────────────────

def step18_qr_verify(image_path_or_array, ocr_fields):
    """
    Step 18 — QR Code Extraction & Cross-Verification.

    Supports Secure QR V2-V5 (post-2019) and Legacy XML/pipe/JSON.
    Cross-checks: name, dob, gender, PIN, district, state.

    Returns qr_result dict.
    """
    section("18 — QR Code Cross-Verification")
    info("Scanning image for QR code (upscale + 50+ preprocessing variants)...")

    # Load image
    if isinstance(image_path_or_array, str):
        img_bgr = cv2.imread(image_path_or_array)
        if img_bgr is None:
            warn("Cannot load image for QR scan")
    else:
        img_bgr = image_path_or_array

    qr_result = {
        'qr_found':         False,
        'qr_decoded':       False,
        'qr_raw':           None,
        'qr_fields':        {},
        'qr_detect_source': None,
        'qr_format':        None,
        'field_checks':     {},
        'trust_score':      0,
        'verdict':          'UNDETERMINED',
        'fraud_signals':    [],
    }

    if img_bgr is None:
        warn("No image available for QR scan — skipping")
        qr_result['verdict'] = 'UNDETERMINED (no image)'
        return qr_result

    # Try QR detection
    raw_qr = None
    source = None

    pyzbar_results = _detect_qr_pyzbar(img_bgr)
    if pyzbar_results:
        raw_qr, _, variant_label = pyzbar_results[0]
        source = f"pyzbar [{variant_label}]"
        ok(f"QR detected by pyzbar ({len(raw_qr)} bytes) variant={variant_label}")

    if not raw_qr:
        raw_qr, _, variant_label = _detect_qr_opencv(img_bgr)
        if raw_qr:
            source = f"OpenCV [{variant_label}]"
            ok(f"QR detected by OpenCV ({len(raw_qr)} bytes) variant={variant_label}")

    # QR not found -> OCR-compensated score
    if not raw_qr:
        warn("No QR code detected after 50+ preprocessing variants")
        warn("  * Make sure the back side of the card is provided")
        warn("  * Image needs >= 150 DPI for dense Secure QR")
        warn("  * pip install pyzbar  +  sudo apt install libzbar0")

        ocr_score = 10
        aadhaar = ocr_fields.get('aadhaar_number', '')
        if aadhaar and re.match(r'^[2-9]\d{3}\s\d{4}\s\d{4}$', aadhaar.strip()):
            ocr_score += 25
            info("  OCR Aadhaar# format valid -> +25")
        name = ocr_fields.get('name', '')
        if name and len(name.strip()) >= 5 and re.match(r"^[A-Za-z\s.'\-]+$", name.strip()):
            ocr_score += 15
            info("  OCR Name plausible -> +15")
        dob = ocr_fields.get('dob', '')
        if dob and re.match(r'^\d{2}/\d{2}/\d{4}$', dob.strip()):
            try:
                d_v, m_v, y_v = dob.strip().split('/')
                if 1 <= int(d_v) <= 31 and 1 <= int(m_v) <= 12 and 1900 <= int(y_v) <= 2025:
                    ocr_score += 10
                    info("  OCR DOB valid -> +10")
            except ValueError:
                pass
        gender = ocr_fields.get('gender', '')
        if gender and gender.strip().capitalize() in ('Male', 'Female', 'Transgender'):
            ocr_score += 5
            info("  OCR Gender valid -> +5")

        ocr_score = min(ocr_score, 70)
        info(f"  OCR-compensated trust score: {ocr_score}/70")
        qr_result['fraud_signals'].append(
            "QR not detected — score based on OCR field validation only")
        qr_result['trust_score'] = ocr_score
        qr_result['verdict'] = (
            'REVIEW REQUIRED (QR unreadable — OCR fields valid)'
            if ocr_score >= 55 else 'REVIEW REQUIRED (QR not found)')
        return qr_result

    qr_result['qr_found']         = True
    qr_result['qr_raw']           = (raw_qr.decode('utf-8', errors='replace')
                                     if isinstance(raw_qr, bytes) else raw_qr)
    qr_result['qr_detect_source'] = source

    # Parse QR payload
    info("Parsing QR payload...")
    qr_fields = _parse_aadhaar_qr(raw_qr)
    qr_result['qr_fields'] = qr_fields

    if not qr_fields:
        warn("QR found but payload could not be parsed")
        qr_result['fraud_signals'].append(
            "QR present but unreadable — structure may be altered")
        qr_result['trust_score'] = 40
        qr_result['verdict']     = 'REVIEW REQUIRED (QR unreadable)'
        return qr_result

    qr_result['qr_decoded'] = True
    qr_result['qr_format']  = qr_fields.get('_format', 'unknown')
    ok(f"QR decoded: format={qr_result['qr_format']}  "
       f"name='{qr_fields.get('name','')}' "
       f"dob='{qr_fields.get('dob','')}' "
       f"pin='{qr_fields.get('pin','')}'")

    # Cross-check all fields
    print()
    info("Cross-checking QR fields against OCR fields...")
    print()

    trust_score   = 10 + 10   # QR found + decoded
    field_checks  = {}
    fraud_signals = []

    def _check(label, pts, match, note, fraud_msg=None):
        nonlocal trust_score
        field_checks[label] = {'match': match, 'note': note}
        if match is True:
            trust_score += pts
            ok(f"  {label:<18} [MATCH  +{pts}pt] : {note}")
        elif match is False:
            print(f"  [XX]  {label:<18} [MISMATCH] : {note}")
            fraud_signals.append(fraud_msg or f"{label} mismatch — {note}")
        else:
            info(f"  {label:<18} [SKIP      ] : {note}")

    # Name (+20)
    if qr_fields.get('name') and ocr_fields.get('name'):
        nm, _, nn = _names_match(ocr_fields['name'], qr_fields['name'])
        _check('name', 20, nm, nn,
               f"Name mismatch — OCR:'{ocr_fields['name']}' QR:'{qr_fields['name']}'")
    else:
        _check('name', 20, None, 'not available')

    # DOB (+20)
    if qr_fields.get('dob') and ocr_fields.get('dob'):
        dm, dn = _dobs_match(ocr_fields['dob'], qr_fields['dob'])
        _check('dob', 20, dm, dn,
               f"DOB mismatch — OCR:'{ocr_fields['dob']}' QR:'{qr_fields['dob']}'")
    else:
        _check('dob', 20, None, 'not available')

    # Gender (+10)
    if qr_fields.get('gender') and ocr_fields.get('gender'):
        qg = qr_fields['gender'].capitalize()
        og = ocr_fields['gender'].capitalize()
        gm = (qg == og)
        _check('gender', 10, gm,
               'match' if gm else f"OCR='{og}' QR='{qg}'")
    else:
        _check('gender', 10, None, 'not available')

    # PIN (+10)
    if qr_fields.get('pin') and ocr_fields.get('address_pin'):
        pm, pn = _pins_match(ocr_fields['address_pin'], qr_fields['pin'])
        _check('pin', 10, pm, pn,
               f"PIN mismatch — OCR:'{ocr_fields['address_pin']}' QR:'{qr_fields['pin']}'")
    else:
        _check('pin', 10, None, 'not available in QR or OCR')

    # District (+10)
    qr_dist  = qr_fields.get('district') or qr_fields.get('district2', '')
    ocr_dist = ocr_fields.get('address_district', '')
    if qr_dist and ocr_dist:
        lm, ln = _location_match(ocr_dist, qr_dist)
        _check('district', 10, lm, ln,
               f"District mismatch — OCR:'{ocr_dist}' QR:'{qr_dist}'")
    else:
        _check('district', 10, None, 'not available in QR or OCR')

    # State (+10)
    if qr_fields.get('state') and ocr_fields.get('address_state'):
        sm, sn = _location_match(ocr_fields['address_state'], qr_fields['state'])
        _check('state', 10, sm, sn,
               f"State mismatch — OCR:'{ocr_fields['address_state']}' QR:'{qr_fields['state']}'")
    else:
        _check('state', 10, None, 'not available in QR or OCR')

    # Aadhaar# — only meaningful for legacy XML QR
    aadhaar_match, aadhaar_note = _aadhaar_nums_match(
        ocr_fields.get('aadhaar_number'), qr_fields)
    field_checks['aadhaar_number'] = {'match': aadhaar_match, 'note': aadhaar_note}
    if aadhaar_match is True:
        ok(f"  {'aadhaar_number':<18} [MATCH     ] : {aadhaar_note}")
    elif aadhaar_match is False:
        warn(f"  {'aadhaar_number':<18} [INFO      ] : {aadhaar_note}")
    else:
        info(f"  {'aadhaar_number':<18} [SKIP      ] : {aadhaar_note}")

    # Clamp & verdict
    trust_score = min(trust_score, 100)
    qr_result['trust_score']   = trust_score
    qr_result['field_checks']  = field_checks
    qr_result['fraud_signals'] = fraud_signals

    verdict = ("LIKELY GENUINE ✓"   if trust_score >= 80 else
               "REVIEW REQUIRED ⚠"  if trust_score >= 50 else
               "FRAUD SUSPECTED ✗")
    qr_result['verdict'] = verdict

    # Summary printout
    print()
    print(f"  {'─'*56}")
    print(f"  QR Format      : {qr_result['qr_format']}")
    print(f"  Trust Score    : {trust_score}/100")
    print(f"  Verdict        : {verdict}")
    print(f"  Detect source  : {source}")
    print()
    print("  ── All QR fields decoded ──")
    display_keys = [
        ('name',          'Name'),
        ('dob',           'Date of Birth'),
        ('gender',        'Gender'),
        ('building',      'Building'),
        ('street',        'Street'),
        ('landmark',      'Landmark'),
        ('locality',      'Locality'),
        ('vtc',           'VTC / Sub-district'),
        ('subdistrict',   'Sub-district'),
        ('district',      'District'),
        ('state',         'State'),
        ('pin',           'PIN Code'),
        ('care_of',       'Care Of (C/O)'),
        ('mobile_masked', 'Mobile (masked)'),
        ('mobile_last4',  'Mobile last 4'),
        ('reference_id',  'Reference ID'),
        ('qr_version',    'QR Version'),
    ]
    for key, label in display_keys:
        val = qr_fields.get(key, '')
        if val:
            print(f"  {label:<22}: {val}")

    if fraud_signals:
        print()
        print(f"  Fraud Signals ({len(fraud_signals)}):")
        for sig in fraud_signals:
            print(f"    * {sig}")
    print(f"  {'─'*56}")

    return qr_result