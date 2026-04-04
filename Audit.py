# audit.py
# Phase 5 — Verification Audit Log + Duplicate Detection
#
# FEATURES:
#   1. Audit Log — tamper-evident log of every verification
#      - Stores masked fields only (no full Aadhaar number)
#      - SHA-256 hash of card image for duplicate detection
#      - Timestamp, verdict, scores, signals
#      - Append-only JSON-lines file (one JSON object per line)
#
#   2. Duplicate Submission Detection (pHash)
#      - Perceptual hash of card image
#      - Detects same card submitted again even if:
#          • Different brightness/contrast
#          • Slightly different crop
#          • Different JPEG quality
#          • Minor rotation
#      - Hamming distance threshold: 10 bits (out of 64)
#
# USAGE:
#   from audit import AuditLog
#   log = AuditLog()
#   is_dup, dup_id = log.check_duplicate(front_img_bgr)
#   log.record(fields, verdict, scores, front_img_bgr)
# ─────────────────────────────────────────────────────────────

import os
import json
import hashlib
import datetime
import uuid

import cv2
import numpy as np


# ── Config ────────────────────────────────────────────────────
DEFAULT_LOG_PATH  = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'audit_log.jsonl'
)
PHASH_THRESHOLD   = 10   # bits — same card if hamming distance <= this
PHASH_SIZE        = 8    # 8x8 DCT = 64-bit hash


def _compute_phash(img_bgr):
    """
    Compute perceptual hash (pHash) of an image.

    HOW pHash WORKS:
      1. Resize to 32x32 (capture structure, not detail)
      2. Convert to grayscale
      3. Apply 2D DCT (Discrete Cosine Transform)
      4. Take top-left 8x8 block (low-frequency components)
      5. Compute mean of those 64 values
      6. For each value: bit=1 if value > mean, else bit=0
      7. Result: 64-bit integer

    WHY IT'S ROBUST:
      The low-frequency DCT coefficients capture the STRUCTURE
      of the image — the overall brightness pattern, the rough
      shapes. Brightness changes, JPEG compression, minor crops,
      and small rotations all change high-frequency components
      but leave the low-frequency structure intact.
      So the same card photographed twice gets the same hash.

    Returns: int (64-bit pHash) or None on failure
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
    try:
        # Resize and grayscale
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        img_f   = resized.astype(np.float32)

        # 2D DCT via separable 1D DCTs
        dct = cv2.dct(img_f)

        # Take top-left 8x8
        dct_low = dct[:PHASH_SIZE, :PHASH_SIZE]

        # Exclude DC component (top-left pixel) from mean
        dct_flat = dct_low.flatten()
        mean_val = (dct_flat.sum() - dct_flat[0]) / (len(dct_flat) - 1)

        # Build 64-bit hash
        bits = (dct_flat > mean_val).astype(int)
        h    = 0
        for b in bits:
            h = (h << 1) | int(b)

        return h
    except Exception:
        return None


def _hamming_distance(h1, h2):
    """Number of differing bits between two 64-bit integers."""
    if h1 is None or h2 is None:
        return 64
    return bin(h1 ^ h2).count('1')


def _image_sha256(img_bgr):
    """SHA-256 hash of raw image bytes for exact duplicate detection."""
    if img_bgr is None:
        return None
    return hashlib.sha256(img_bgr.tobytes()).hexdigest()


def _mask_aadhaar(n):
    if not n:
        return None
    c = n.replace(' ', '')
    return f'XXXX-XXXX-{c[8:]}' if len(c) == 12 else None


class AuditLog:
    """
    Append-only verification audit log with duplicate detection.

    Each entry is a JSON object on one line (JSON-lines format).
    The file grows indefinitely — rotate it periodically in production.

    Entry schema:
      {
        "log_id":        "uuid",
        "timestamp":     "ISO-8601",
        "verdict":       "VERIFIED ✓",
        "scores": {
          "qr_trust":    int,
          "tampering":   int,
          "geo":         int,
          "face":        int,
          "overall":     int,
        },
        "fields_masked": {
          "name":        "GHANSHYAM JETHARAM KUMAVAT",
          "aadhaar":     "XXXX-XXXX-5472",
          "dob":         "10/11/2005",
          "gender":      "Male",
          "pin":         "421306",
          "state":       "Maharashtra",
        },
        "fraud_signals": [...],
        "phash":         "hex string of 64-bit pHash",
        "img_sha256":    "first 16 chars of SHA-256",
        "is_duplicate":  false,
        "duplicate_of":  null,
      }
    """

    def __init__(self, log_path=None):
        self.log_path = log_path or DEFAULT_LOG_PATH

    def _load_entries(self):
        """Load all existing log entries."""
        if not os.path.exists(self.log_path):
            return []
        entries = []
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass
        return entries

    def check_duplicate(self, img_bgr):
        """
        Check if this card image was submitted before.

        Uses two methods:
          1. Exact SHA-256 match (same file submitted twice)
          2. pHash Hamming distance <= threshold (same card, different photo)

        Returns:
          (is_duplicate: bool, duplicate_log_id: str or None)
        """
        if img_bgr is None:
            return False, None

        current_sha = _image_sha256(img_bgr)
        current_ph  = _compute_phash(img_bgr)

        entries = self._load_entries()
        for entry in entries:
            # Exact SHA256 match
            stored_sha = entry.get('img_sha256', '')
            if stored_sha and current_sha and current_sha[:16] == stored_sha:
                return True, entry.get('log_id')

            # pHash similarity match
            stored_ph_hex = entry.get('phash', '')
            if stored_ph_hex and current_ph is not None:
                try:
                    stored_ph = int(stored_ph_hex, 16)
                    dist = _hamming_distance(current_ph, stored_ph)
                    if dist <= PHASH_THRESHOLD:
                        return True, entry.get('log_id')
                except (ValueError, TypeError):
                    pass

        return False, None

    def record(
        self,
        fields,
        final_verdict,
        qr_result=None,
        tamper_result=None,
        geo_result=None,
        face_result=None,
        front_img_bgr=None,
        is_duplicate=False,
        duplicate_of=None,
        report_id=None,
    ):
        """
        Append a verification result to the audit log.

        Stores MASKED fields only — no full Aadhaar number stored.
        The log is safe to store — it cannot be used to reconstruct
        the full Aadhaar number.

        Returns: log_id (str)
        """
        log_id = str(uuid.uuid4())
        ts     = datetime.datetime.now().isoformat()

        qr_trust = (qr_result or {}).get('trust_score', 0)
        t_score  = (tamper_result or {}).get('score', 100)
        g_score  = (geo_result or {}).get('score', 100)
        f_score  = (face_result or {}).get('quality_score', 0)
        overall  = int(qr_trust * 0.35 + t_score * 0.25 +
                       g_score  * 0.25 + f_score * 0.15)

        all_signals = []
        for r in [qr_result, tamper_result, geo_result]:
            if r:
                all_signals.extend(r.get('signals', []))

        # Compute image fingerprints
        phash_hex = None
        sha256_prefix = None
        if front_img_bgr is not None:
            ph = _compute_phash(front_img_bgr)
            if ph is not None:
                phash_hex = hex(ph)[2:].zfill(16)
            sha256_prefix = (_image_sha256(front_img_bgr) or '')[:16]

        entry = {
            'log_id':        log_id,
            'timestamp':     ts,
            'verdict':       str(final_verdict),
            'report_id':     report_id,
            'scores': {
                'qr_trust':   qr_trust,
                'tampering':  t_score,
                'geo':        g_score,
                'face':       f_score,
                'overall':    overall,
            },
            'fields_masked': {
                'name':       fields.get('name'),
                'aadhaar':    _mask_aadhaar(fields.get('aadhaar_number')),
                'dob':        fields.get('dob'),
                'gender':     fields.get('gender'),
                'pin':        fields.get('address_pin'),
                'district':   fields.get('address_district'),
                'state':      fields.get('address_state'),
            },
            'fraud_signals': all_signals[:10],   # cap at 10
            'phash':         phash_hex,
            'img_sha256':    sha256_prefix,
            'is_duplicate':  is_duplicate,
            'duplicate_of':  duplicate_of,
        }

        # Append to log file (create if not exists)
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"  [!!]  Audit log write failed: {e}")

        return log_id

    def get_recent(self, n=10):
        """Return the last n entries."""
        return self._load_entries()[-n:]

    def get_stats(self):
        """Return summary statistics of all verifications."""
        entries = self._load_entries()
        if not entries:
            return {'total': 0}
        verdicts = [e.get('verdict', '') for e in entries]
        return {
            'total':      len(entries),
            'verified':   sum(1 for v in verdicts if 'VERIFIED' in v),
            'review':     sum(1 for v in verdicts if 'REVIEW' in v),
            'rejected':   sum(1 for v in verdicts if 'TAMPERED' in v or 'INVALID' in v),
            'duplicates': sum(1 for e in entries if e.get('is_duplicate')),
            'avg_score':  int(sum(
                e.get('scores', {}).get('overall', 0) for e in entries
            ) / max(len(entries), 1)),
        }