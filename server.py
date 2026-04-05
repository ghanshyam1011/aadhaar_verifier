# server.py — AadhaarCheck v5
# Phase 5: PDF report, audit log, duplicate detection, quality pre-check
# Phase 6: API key auth, rate limiting, auto-delete, structured errors
#
# ENDPOINTS:
#   POST /api/validate      — main verification
#   GET  /api/report/<id>   — download PDF report
#   GET  /api/stats         — audit statistics
#   GET  /api/health        — health check
#
# ENV VARS (.env file):
#   API_KEYS=key1,key2       — comma-separated valid keys
#   REQUIRE_API_KEY=false    — set true in production
# ─────────────────────────────────────────────────────────────

import os, time, shutil, tempfile, datetime, re
from collections import defaultdict

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

from main import run_pipeline

# ── Make sure THIS file's folder is always on sys.path ───────
# Fixes "No module named 'report_generator'" when Flask is
# launched from a different working directory (common on Windows).
import sys as _sys, os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

# Pre-import report_generator so errors surface immediately at startup
# rather than silently during the first request.
try:
    from report_generator import generate_pdf_report as _gen_pdf
    _PDF_AVAILABLE = True
except ImportError as _e:
    print(f"  [!!]  report_generator not available: {_e}")
    print(f"        Make sure report_generator.py is in: {_HERE}")
    _gen_pdf = None
    _PDF_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__, static_folder=".", template_folder=".")
CORS(app)

# ── Config ────────────────────────────────────────────────────
REQUIRE_API_KEY    = os.environ.get('REQUIRE_API_KEY', 'false').lower() == 'true'
API_KEYS           = set(k.strip() for k in
                         os.environ.get('API_KEYS', '').split(',') if k.strip())
RATE_LIMIT_MIN     = 10
RATE_LIMIT_DAY     = 100
_rate_store        = defaultdict(list)
_report_store      = {}   # report_id → pdf_path
_audit_log         = None


def _get_audit():
    global _audit_log
    if _audit_log is None:
        try:
            from audit import AuditLog
            _audit_log = AuditLog()
        except Exception:
            pass
    return _audit_log


def _cleanup(temp_dir):
    try:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


def _to_python(obj):
    """
    Recursively convert any object to a JSON-safe Python primitive.
    Handles ALL numpy scalar types using numpy's own dtype checking,
    which works across numpy 1.x and 2.x on all platforms.
    """
    # None → null
    if obj is None:
        return None

    # numpy array → list (recurse each element)
    if isinstance(obj, np.ndarray):
        return [_to_python(x) for x in obj.tolist()]

    # numpy scalar — use numpy's own item() to convert to Python native
    # This handles np.bool_, np.int8/16/32/64, np.uint*, np.float16/32/64
    # np.complex*, np.str_, np.bytes_ — everything numpy defines
    if isinstance(obj, np.generic):
        py = obj.item()           # always returns a Python native type
        if isinstance(py, bool):  return bool(py)
        if isinstance(py, int):   return int(py)
        if isinstance(py, float): return float(py)
        return py                 # str, bytes, complex etc.

    # Python bool BEFORE int (bool is subclass of int in Python)
    if isinstance(obj, bool):
        return bool(obj)

    # Python set → list
    if isinstance(obj, set):
        return [_to_python(x) for x in obj]

    # dict — recurse keys and values
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}

    # list / tuple — recurse
    if isinstance(obj, (list, tuple)):
        return [_to_python(x) for x in obj]

    # Python int / float — cast to make sure no subclasses sneak through
    if isinstance(obj, int):   return int(obj)
    if isinstance(obj, float): return float(obj)

    # bytes → base64 string (safe for JSON)
    if isinstance(obj, bytes):
        import base64
        return base64.b64encode(obj).decode('ascii')

    # Everything else: str, bool already handled above, pass through
    return obj


def _convert(obj):
    """Public alias used throughout the server."""
    return _to_python(obj)


def _rate_ok(key):
    now = time.time()
    _rate_store[key] = [t for t in _rate_store[key] if now - t < 86400]
    recent = [t for t in _rate_store[key] if now - t < 60]
    if len(recent) >= RATE_LIMIT_MIN:
        return False, f"Rate limit: max {RATE_LIMIT_MIN} requests/minute"
    if len(_rate_store[key]) >= RATE_LIMIT_DAY:
        return False, f"Rate limit: max {RATE_LIMIT_DAY} requests/day"
    _rate_store[key].append(now)
    return True, "ok"


def _auth():
    if not REQUIRE_API_KEY:
        return True, request.remote_addr or 'anon', None
    key = request.headers.get('X-API-Key', '').strip()
    if not key:
        return False, None, (jsonify({'error': 'Missing X-API-Key header'}), 401)
    if key not in API_KEYS:
        return False, None, (jsonify({'error': 'Invalid API key'}), 403)
    return True, key, None


def _quality_check(img_path, label='Image'):
    img = cv2.imread(img_path)
    if img is None:
        return False, f"{label}: cannot read file — check format (JPG/PNG/WEBP)"
    h, w = img.shape[:2]
    if w < 300 or h < 200:
        return False, f"{label}: too small ({w}x{h}px) — minimum 300x200px needed"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < 30:
        return False, f"{label}: too dark (brightness={brightness:.0f}) — use better lighting"
    if brightness > 248:
        return False, f"{label}: appears blank/white — check the file"
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < 5:
        return False, f"{label}: completely blurred or solid colour — unusable"
    aspect = w / h
    if not (0.4 <= aspect <= 3.5):
        return False, f"{label}: unusual aspect ratio ({aspect:.2f}) — is this a card photo?"
    return True, "ok"


def _plain_english(result):
    explanations = []
    for src_key, label in [('qr','QR'), ('tamper','Tampering'), ('geo','Geo')]:
        for sig in (result.get(src_key) or {}).get('signals', []):
            sl = sig.lower()
            if 'verhoeff' in sl or 'checksum' in sl:
                plain = ('The Aadhaar number fails its mathematical check-digit '
                         '(Verhoeff algorithm). Every real Aadhaar number must pass '
                         'this test — this one does not. The number is likely fabricated.')
                sev = 'critical'
            elif 'ela' in sl or 're-compression' in sl:
                plain = ('A region of the card image was re-compressed at a different '
                         'JPEG quality level — a sign that text or numbers were edited '
                         'in photo software before submission.')
                sev = 'critical'
            elif 'moire' in sl or 'screen' in sl:
                plain = ('The card appears to be a photo of a screen or a photocopy. '
                         'Periodic interference patterns were detected. '
                         'Please photograph the original physical card.')
                sev = 'critical'
            elif 'pin' in sl and 'belongs to' in sl:
                plain = ('The PIN code does not match the stated state. '
                         'Every Indian PIN code maps to exactly one state — '
                         'this combination is geographically impossible.')
                sev = 'critical'
            elif 'district' in sl and 'mismatch' in sl:
                plain = ('The district does not belong to the stated state. '
                         'This address combination does not exist in India.')
                sev = 'critical'
            elif 'age' in sl and 'mismatch' in sl:
                plain = ('The estimated age of the person in the photo does not '
                         'match the date of birth on the card. This may indicate '
                         'a stolen card used by a different person.')
                sev = 'warning'
            elif 'noise' in sl or 'clone' in sl:
                plain = ('Sensor noise analysis found an anomalous region. '
                         'Part of the image may have been copied and pasted '
                         'from another photo.')
                sev = 'warning'
            elif 'ai' in sl and 'generat' in sl:
                plain = ('The image shows characteristics of AI-generated content. '
                         'The DCT coefficient distribution matches patterns from '
                         'tools like Stable Diffusion or Midjourney.')
                sev = 'critical'
            elif 'qr not detected' in sl:
                plain = ('The QR code could not be scanned. This may be because '
                         'the back side was not provided, the QR is damaged, '
                         'or the image resolution is too low.')
                sev = 'warning'
            else:
                plain = sig
                sev   = 'warning'
            explanations.append({'source': label, 'signal': sig,
                                  'plain': plain, 'severity': sev})
    return explanations


# ── Routes ────────────────────────────────────────────────────

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory('.', path)


@app.route('/api/health')
def health():
    audit = _get_audit()
    return jsonify({
        'status':    'ok',
        'timestamp': datetime.datetime.now().isoformat(),
        'version':   'AadhaarCheck v5',
        'stats':     audit.get_stats() if audit else {},
    })


@app.route('/api/stats')
def stats():
    audit = _get_audit()
    return jsonify(audit.get_stats() if audit else {'error': 'audit not available'})


@app.route('/api/report/<report_id>')
def download_report(report_id):
    pdf_path = _report_store.get(report_id)
    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({'error': 'Report not found or expired'}), 404
    return send_file(pdf_path, mimetype='application/pdf',
                     as_attachment=True,
                     download_name=f'aadhaar_{report_id}.pdf')


@app.route('/api/validate', methods=['POST'])
def validate():
    temp_dir = None
    try:
        # Auth
        ok, rate_key, err = _auth()
        if not ok:
            return err

        # Rate limit
        allowed, reason = _rate_ok(rate_key)
        if not allowed:
            return jsonify({'error': reason}), 429

        # Files
        front  = request.files.get('front')
        back   = request.files.get('back')
        selfie = request.files.get('selfie')

        if not front:
            return jsonify({
                'error': 'Front image required',
                'plain': 'Please upload a photo of the front side of the Aadhaar card.'
            }), 400

        # Save to temp
        temp_dir   = tempfile.mkdtemp()
        front_path = os.path.join(temp_dir, 'front.jpg')
        front.save(front_path)

        back_path = selfie_path = None
        if back and back.filename:
            back_path = os.path.join(temp_dir, 'back.jpg')
            back.save(back_path)
        if selfie and selfie.filename:
            selfie_path = os.path.join(temp_dir, 'selfie.jpg')
            selfie.save(selfie_path)

        # Quality pre-check
        for path, label in [(front_path, 'Front image'),
                            (back_path,  'Back image') if back_path else (None, None)]:
            if not path:
                continue
            qok, qreason = _quality_check(path, label)
            if not qok:
                return jsonify({'error': qreason,
                                'plain': qreason}), 400

        # Duplicate check
        front_img  = cv2.imread(front_path)
        audit      = _get_audit()
        is_dup, dup_id = False, None
        if audit and front_img is not None:
            is_dup, dup_id = audit.check_duplicate(front_img)
        if is_dup:
            return jsonify({
                'error':        'Duplicate submission',
                'plain':        ('This card has already been submitted for verification. '
                                 'Duplicate submissions are blocked to prevent fraud.'),
                'duplicate_of': dup_id,
            }), 409

        # Run pipeline
        result = run_pipeline(front_path, back_path, selfie_path)

        # PDF report
        report_id = report_hash = None
        try:
            if not _PDF_AVAILABLE:
                raise ImportError("report_generator.py not found — skipping PDF")
            generate_pdf_report = _gen_pdf
            fields        = result.get('fields', {})
            verdict_str   = result.get('verdict', {}).get('label', 'REVIEW REQUIRED')
            verification  = {}
            for lbl, key, pat in [
                ('Name',           'name',           r'.{3,}'),
                ('Date of Birth',  'dob',            r'\d{2}/\d{2}/\d{4}'),
                ('Gender',         'gender',         r'Male|Female|Transgender'),
                ('Aadhaar Number', 'aadhaar_number', r'[2-9]\d{3}\s\d{4}\s\d{4}'),
            ]:
                v = fields.get(key, '')
                ok_ = bool(v and re.match(pat, str(v)))
                verification[lbl] = {'value': v, 'valid': ok_,
                                     'msg': 'Valid ✓' if ok_ else 'Not found'}

            pdf_path, report_id, report_hash = generate_pdf_report(
                fields=fields, verification=verification,
                final_verdict=verdict_str,
                qr_result=result.get('qr'),
                tamper_result=result.get('tamper'),
                geo_result=result.get('geo'),
                face_result=result.get('face'),
            )
            _report_store[report_id] = pdf_path
        except Exception as e:
            print(f"  [!!]  PDF failed: {e}")

        # Audit log
        if audit:
            audit.record(
                fields=result.get('fields', {}),
                final_verdict=result.get('verdict', {}).get('label', ''),
                qr_result=result.get('qr'),
                tamper_result=result.get('tamper'),
                geo_result=result.get('geo'),
                face_result=result.get('face'),
                front_img_bgr=front_img,
                report_id=report_id,
            )

        # Add to response
        result['explanations']    = _plain_english(result)
        result['pdf_report_id']   = report_id
        result['pdf_report_hash'] = report_hash
        result['pdf_report_url']  = f'/api/report/{report_id}' if report_id else None

        return jsonify(_convert(result))

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': 'Pipeline failed', 'details': str(e)}), 500

    finally:
        _cleanup(temp_dir)   # always delete uploaded images


if __name__ == '__main__':
    print(f"AadhaarCheck v5 | Auth={'ON' if REQUIRE_API_KEY else 'OFF'} | "
          f"Rate={RATE_LIMIT_MIN}/min {RATE_LIMIT_DAY}/day | Auto-delete=ON")
    app.run(host='0.0.0.0', port=5000, debug=True)