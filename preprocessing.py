# preprocessing.py
# Steps 1–12: Load, Resize, SuperResolution, Orient, MaskPhoto,
#             RemoveColorNoise, Grayscale, Denoise, CLAHE,
#             AdaptiveSharpen, Binarize, Deskew, MorphCleanup
# ─────────────────────────────────────────────────────────────
# No code changed — same functions, split into own file.
# Imports shared helpers from utils.py

import cv2
import numpy as np
import os
import sys

from utils import section, ok, info, warn, err

# ─────────────────────────────────────────────────────────────
#  STEP 1 — Load
# ─────────────────────────────────────────────────────────────

def step1_load(path):
    section("1 — Load Image")
    if not os.path.exists(path):
        err(f"File not found: {path}"); sys.exit(1)
    img = cv2.imread(path)
    if img is None:
        err("Cannot read image."); sys.exit(1)
    h, w = img.shape[:2]
    ok(f"Loaded : {path}")
    ok(f"Size   : {w} x {h} px  |  {os.path.getsize(path)//1024} KB")
    return img


# ─────────────────────────────────────────────────────────────
#  STEP 2 — Resize to 1600px wide (better for Tesseract)
# ─────────────────────────────────────────────────────────────

def step2_resize(img, target_w=1600):
    section("2 — Resize")
    h, w = img.shape[:2]
    scale   = target_w / w
    new_h   = int(h * scale)
    method  = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(img, (target_w, new_h), interpolation=method)
    ok(f"{w}x{h}  →  {target_w}x{new_h}  (scale={scale:.2f}x)")
    return resized



# ─────────────────────────────────────────────────────────────
#  STEP 2B — Super Resolution (Real-ESRGAN)
#
#  WHY THIS HELPS:
#    Low-res or blurry Aadhaar card photos (< 150 DPI) cause
#    Tesseract and PaddleOCR to fail on fine text details.
#    Real-ESRGAN is a deep learning model that:
#      1. Upscales the image 2x or 4x
#      2. Simultaneously denoises and sharpens
#      3. Reconstructs compressed text strokes
#    Result: 30-60% OCR accuracy improvement on blurry images.
#
#  MODEL FILE (download once):
#    RealESRGAN_x2plus.pth — place in same folder as this script
#    https://github.com/xinntao/Real-ESRGAN/releases
#
#  INSTALL:
#    pip install realesrgan basicsr
#
#  GRACEFUL FALLBACK:
#    If not installed or weights missing — silently skipped.
#    Script never crashes due to missing SR dependencies.
# ─────────────────────────────────────────────────────────────

def step2b_super_resolution(img, weights_path=None):
    section("2B — Super Resolution (Real-ESRGAN)")

    # ── Auto-locate weights file ──────────────────────────
    if weights_path is None:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()  # Colab/Jupyter: no __file__
        for name in ["RealESRGAN_x2plus.pth", "RealESRGAN_x4plus.pth", "RealESRGAN_x2.pth"]:
            candidate = os.path.join(script_dir, name)
            if os.path.exists(candidate):
                weights_path = candidate
                break

    # ── Check dependencies ────────────────────────────────
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import torch
    except ImportError:
        warn("realesrgan not installed — skipping Super Resolution")
        warn("To enable: pip install realesrgan basicsr")
        info("Continuing with standard preprocessing...")
        return img

    if not weights_path or not os.path.exists(weights_path):
        warn("Real-ESRGAN weights not found — skipping Super Resolution")
        warn("Download: RealESRGAN_x2plus.pth → place in same folder as script")
        info("Link: https://github.com/xinntao/Real-ESRGAN/releases")
        info("Continuing with standard preprocessing...")
        return img

    # ── Detect scale from weights filename ───────────────
    import torch
    scale  = 4 if "x4" in os.path.basename(weights_path) else 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Weights : {os.path.basename(weights_path)}")
    info(f"Scale   : {scale}x upscale")
    ok(f"Device  : {device}  ({'GPU' if str(device)=='cuda' else 'CPU — may be slow on large images'})")

    # ── Load model ────────────────────────────────────────
    info("Loading Real-ESRGAN model...")
    try:
        model_arch = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=scale
        )
        upsampler = RealESRGANer(
            scale=scale,
            model_path=weights_path,
            model=model_arch,
            tile=400,       # tile mode prevents out-of-memory on large images
            tile_pad=10,
            pre_pad=0,
            half=False,     # half=True only with CUDA GPU
            device=device,
        )
        ok("Model loaded successfully")
    except Exception as e:
        warn(f"Failed to load model: {e}")
        info("Continuing without Super Resolution...")
        return img

    # ── Run super resolution ──────────────────────────────
    h, w = img.shape[:2]
    info(f"Input resolution  : {w} x {h} px")
    try:
        sr_img, _ = upsampler.enhance(img, outscale=scale)
        sh, sw    = sr_img.shape[:2]
        ok(f"Output resolution : {sw} x {sh} px  ({scale}x)")

        # Measure sharpness improvement
        b_score = cv2.Laplacian(cv2.cvtColor(img,    cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        a_score = cv2.Laplacian(cv2.cvtColor(sr_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        gain    = ((a_score - b_score) / max(b_score, 1)) * 100
        info(f"Sharpness: {b_score:.1f}  ->  {a_score:.1f}  (improvement: +{gain:.0f}%)")
        ok("Super Resolution complete!")
        return sr_img
    except Exception as e:
        warn(f"Super resolution failed during enhance: {e}")
        info("Continuing without Super Resolution...")
        return img

# ─────────────────────────────────────────────────────────────
#  STEP 3 — Auto-orient
# ─────────────────────────────────────────────────────────────

def step3_orient(img):
    section("3 — Auto-Orient")
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        ok(f"Portrait → rotated to landscape")
    else:
        ok(f"Already landscape ({w}x{h})")
    return img


# ─────────────────────────────────────────────────────────────
#  STEP 4 — Remove the photo region (left ~25% of card)
#           The face photo confuses Tesseract heavily
# ─────────────────────────────────────────────────────────────

def step4_mask_photo(img):
    section("4 — Mask Face Photo Region")
    h, w = img.shape[:2]
    masked = img.copy()

    # Aadhaar cards: photo is always in left ~28% of the card
    # and vertically in the middle ~60% of the card
    x_end = int(w * 0.28)
    y_start = int(h * 0.18)
    y_end   = int(h * 0.82)
    masked[y_start:y_end, 0:x_end] = 255   # white out the photo

    ok(f"Whited out photo region: x=0→{x_end}, y={y_start}→{y_end}")
    info("Prevents Tesseract from misreading face features as text")
    return masked


# ─────────────────────────────────────────────────────────────
#  STEP 5 — Remove colored header/footer bars
#           The tricolor (saffron/green) strips confuse threshold
# ─────────────────────────────────────────────────────────────

def step5_remove_color_noise(img):
    section("5 — Remove Colored Header/Footer Strips")
    hsv     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Saffron/Orange mask
    saffron_lo = np.array([5,  80,  80])
    saffron_hi = np.array([25, 255, 255])

    # Green mask
    green_lo   = np.array([40, 50, 50])
    green_hi   = np.array([85, 255, 255])

    mask_s = cv2.inRange(hsv, saffron_lo, saffron_hi)
    mask_g = cv2.inRange(hsv, green_lo,   green_hi)
    mask   = cv2.bitwise_or(mask_s, mask_g)

    # Dilate to fully cover partial color regions
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.dilate(mask, kernel, iterations=2)

    cleaned = img.copy()
    cleaned[mask > 0] = 255   # replace colored areas with white

    ok("Removed saffron/orange regions")
    ok("Removed green regions")
    px_removed = np.sum(mask > 0)
    info(f"Masked out {px_removed:,} pixels of colored noise")
    return cleaned


# ─────────────────────────────────────────────────────────────
#  STEP 6 — Grayscale
# ─────────────────────────────────────────────────────────────

def step6_grayscale(img):
    section("6 — Convert to Grayscale")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok(f"BGR → Grayscale  |  shape: {gray.shape}")
    return gray


# ─────────────────────────────────────────────────────────────
#  STEP 7 — Denoise
# ─────────────────────────────────────────────────────────────

def step7_denoise(gray):
    section("7 — Denoise")
    denoised = cv2.fastNlMeansDenoising(gray, h=12,
                                         templateWindowSize=7,
                                         searchWindowSize=21)
    ok("fastNlMeansDenoising applied  (h=12)")
    return denoised


# ─────────────────────────────────────────────────────────────
#  STEP 8 — CLAHE contrast boost
# ─────────────────────────────────────────────────────────────

def step8_clahe(gray):
    section("8 — CLAHE Contrast Enhancement")
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    ok("CLAHE applied  (clipLimit=3.0, tileGridSize=8x8)")
    info(f"Mean pixel: {np.mean(gray):.1f} -> {np.mean(enhanced):.1f}")
    return enhanced


# ─────────────────────────────────────────────────────────────
#  STEP 9 — Blur Detection + Adaptive Sharpening
#
#  HOW BLUR IS MEASURED (Laplacian Variance method):
#    The Laplacian detects edges via the 2nd derivative.
#    Sharp image  → strong edges → HIGH Laplacian variance
#    Blurry image → soft  edges  → LOW  Laplacian variance
#    blur_score = variance( Laplacian(gray) )
#
#  BLUR THRESHOLDS:
#    score > 300   →  SHARP        — light unsharp mask
#    score 100–300 →  MODERATE     — medium unsharp mask
#    score  30–100 →  BLURRY       — aggressive pipeline
#    score   < 30  →  VERY BLURRY  — Wiener deconvolution
# ─────────────────────────────────────────────────────────────

def measure_blur(gray):
    """Returns (score, label). Higher score = sharper image."""
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if   score > 300: label = "SHARP"
    elif score > 100: label = "MODERATE BLUR"
    elif score >  30: label = "BLURRY"
    else:             label = "VERY BLURRY"
    return score, label


def unsharp_mask(gray, sigma, strength):
    """
    Unsharp mask: result = orig*(1+s) - blurred*s
    Higher sigma    = fixes wider/softer blur
    Higher strength = more aggressive edge boost
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(gray, 1 + strength, blurred, -strength, 0)


def wiener_deconvolution(gray, kernel_size=5, snr=0.02):
    """
    Wiener filter in frequency domain.
    Best for motion blur / out-of-focus images.

    How it works:
      1. DFT the image into frequency domain.
      2. Build a Gaussian PSF (models the blur kernel).
      3. Apply Wiener formula:
            W(f) = H*(f) / ( |H(f)|^2 + SNR )
         H  = DFT of PSF
         H* = complex conjugate of H
      4. Multiply image spectrum by W(f).
      5. Inverse DFT back to spatial domain.

    snr: lower = more aggressive recovery but more noise.
         Good range 0.005 to 0.05.
    """
    img_f = gray.astype(np.float64) / 255.0
    h, w  = img_f.shape

    psf = cv2.getGaussianKernel(kernel_size, -1)
    psf = psf @ psf.T
    psf /= psf.sum()

    psf_pad = np.zeros((h, w), dtype=np.float64)
    ph, pw  = psf.shape
    psf_pad[:ph, :pw] = psf
    psf_pad = np.roll(np.roll(psf_pad, -ph//2, axis=0), -pw//2, axis=1)

    IMG    = np.fft.fft2(img_f)
    PSF    = np.fft.fft2(psf_pad)
    PSF_c  = np.conj(PSF)
    wiener = PSF_c / (PSF * PSF_c + snr)
    result = np.abs(np.fft.ifft2(IMG * wiener))
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def laplacian_sharpen(gray):
    """
    Subtracts the Laplacian from the image.
    Enhances ALL edges at once. Stronger than unsharp mask
    for severely blurred text.
      result = original - Laplacian(original)
    """
    lap    = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap    = np.clip(lap, -128, 128).astype(np.int16)
    result = np.clip(gray.astype(np.int16) - lap, 0, 255)
    return result.astype(np.uint8)


def high_pass_blend(gray, sigma=5, blend=0.5):
    """
    Extracts fine edge detail (high-pass) and adds back.
      high_pass = original - GaussianBlur(original)
      result    = original + blend * high_pass
    """
    blurred   = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    high_pass = cv2.subtract(gray, blurred)
    return cv2.addWeighted(gray, 1.0, high_pass, blend, 0)


def step9_adaptive_sharpen(gray):
    section("9 — Blur Detection + Adaptive Sharpening")

    score, label = measure_blur(gray)
    info(f"Laplacian variance score : {score:.2f}")
    ok(f"Blur level detected      : {label}")
    print()

    if score > 300:
        info("Strategy: SHARP — light unsharp mask only")
        result = unsharp_mask(gray, sigma=1.5, strength=0.6)
        ok("Applied: unsharp_mask(sigma=1.5, strength=0.6)")

    elif score > 100:
        info("Strategy: MODERATE — unsharp mask + high-pass")
        result = unsharp_mask(gray, sigma=2.0, strength=1.2)
        ok("Applied: unsharp_mask(sigma=2.0, strength=1.2)")
        result = high_pass_blend(result, sigma=4, blend=0.3)
        ok("Applied: high_pass_blend(sigma=4, blend=0.3)")

    elif score > 30:
        info("Strategy: BLURRY — strong unsharp + Laplacian + high-pass")
        result = unsharp_mask(gray, sigma=2.5, strength=1.8)
        ok("Applied: unsharp_mask(sigma=2.5, strength=1.8)")
        result = laplacian_sharpen(result)
        ok("Applied: laplacian_sharpen()")
        result = high_pass_blend(result, sigma=5, blend=0.5)
        ok("Applied: high_pass_blend(sigma=5, blend=0.5)")

    else:
        info("Strategy: VERY BLURRY — Wiener deconv + full pipeline")
        warn("FFT-based Wiener filter running — may take a few seconds...")
        result = wiener_deconvolution(gray, kernel_size=7, snr=0.02)
        ok("Applied: wiener_deconvolution(kernel=7, snr=0.02)")
        result = unsharp_mask(result, sigma=2.0, strength=1.5)
        ok("Applied: unsharp_mask(sigma=2.0, strength=1.5)")
        result = laplacian_sharpen(result)
        ok("Applied: laplacian_sharpen()")
        result = high_pass_blend(result, sigma=4, blend=0.4)
        ok("Applied: high_pass_blend(sigma=4, blend=0.4)")

    after_score, after_label = measure_blur(result)
    print()
    info(f"Sharpness before : {score:.2f}  ({label})")
    info(f"Sharpness after  : {after_score:.2f}  ({after_label})")
    gain = ((after_score - score) / max(score, 1)) * 100
    ok(f"Sharpness improved by    : {gain:.1f}%")
    return result


# ─────────────────────────────────────────────────────────────
#  STEP 10 — Binarize (two variants kept separately)
# ─────────────────────────────────────────────────────────────

def step10_binarize(gray):
    section("10 — Binarize (3 variants for multi-pass OCR)")

    # Variant A: Otsu global threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ok(f"Variant A: Otsu global threshold")

    # Variant B: Adaptive (good for local lighting variation)
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21, C=10
    )
    ok("Variant B: Adaptive Gaussian  (block=21, C=10)")

    # Variant C: Blended
    blended = cv2.bitwise_or(otsu, adaptive)
    ok("Variant C: Blended (OR of A and B)")

    white = (np.sum(blended == 255) / blended.size) * 100
    info(f"Blended — White: {white:.1f}%   Black: {100-white:.1f}%")

    return otsu, adaptive, blended


# ─────────────────────────────────────────────────────────────
#  STEP 11 — Deskew
# ─────────────────────────────────────────────────────────────

def step11_deskew(binary):
    section("11 — Deskew")
    coords = np.column_stack(np.where(binary < 127))
    if len(coords) < 10:
        warn("Too few dark pixels — skip deskew")
        return binary
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = 90 + angle
    elif angle > 45: angle = angle - 90
    info(f"Detected tilt: {angle:.3f}°")
    if abs(angle) < 0.3:
        ok("Tilt < 0.3° — no correction needed")
        return binary
    h, w   = binary.shape
    M      = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    result = cv2.warpAffine(binary, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    ok(f"Corrected tilt by {angle:.3f}°")
    return result


# ─────────────────────────────────────────────────────────────
#  STEP 12 — Morphological cleanup
# ─────────────────────────────────────────────────────────────

def step12_morph(binary):
    section("12 — Morphological Cleanup")
    k1 = np.ones((1, 1), np.uint8)
    k2 = np.ones((2, 1), np.uint8)
    out = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k1)
    out = cv2.morphologyEx(out,    cv2.MORPH_OPEN,  k1)
    out = cv2.morphologyEx(out,    cv2.MORPH_CLOSE, k2)
    ok("CLOSE (1x1) → OPEN (1x1) → CLOSE (2x1) applied")
    return out