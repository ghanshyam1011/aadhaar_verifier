# verification_summary.py
# Step 15: Field verification (format checks)
# Step 16: Save stage images to disk
# Step 17: Final summary printout (the full box display)
# ─────────────────────────────────────────────────────────────
# No code changed — same functions, split into own file.

import cv2
import re
import os

from utils import section, ok, info, warn, err



# ─────────────────────────────────────────────────────────────
#  STEP 15 — Verify fields
# ─────────────────────────────────────────────────────────────

def step15_verify(fields):
    section("15 — Verification")

    def v_aadhaar(n):
        if not n: return False, "Not found"
        c = n.replace(" ", "")
        if not re.match(r'^\d{12}$', c): return False, f"Need 12 digits, got {len(c)}"
        if c[0] in "01": return False, "Cannot start with 0 or 1"
        return True, "Valid 12-digit format ✓"

    def v_name(n):
        if not n or len(n.strip()) < 3: return False, "Not found"
        if not re.match(r"^[A-Za-z\s.\'-]+$", n):
            return None, "Non-English chars — review manually"
        return True, "Valid ✓"

    def v_dob(d):
        if not d: return False, "Not found"
        m = re.search(r'\d{4}', d)
        if not m: return False, "No year in value"
        yr = int(m.group())
        if not (1900 <= yr <= 2025): return False, f"Year {yr} out of range"
        return True, "Valid ✓"

    def v_gender(g):
        if not g: return False, "Not found"
        if g.upper() in ["MALE","FEMALE","TRANSGENDER"]: return True, "Valid ✓"
        return None, f"Unusual value: '{g}'"

    checks = [
        ("Name",           fields.get("name"),           v_name),
        ("Date of Birth",  fields.get("dob"),            v_dob),
        ("Gender",         fields.get("gender"),         v_gender),
        ("Aadhaar Number", fields.get("aadhaar_number"), v_aadhaar),
    ]

    results = {}
    all_valid = True
    print()
    for label, value, fn in checks:
        valid, msg = fn(value)
        results[label] = {"value": value, "valid": valid, "msg": msg}
        if valid is True:
            print(f"  [OK]  {label:<22}: {msg}")
        elif valid is False:
            print(f"  [XX]  {label:<22}: ✗  {msg}")
            all_valid = False
        else:
            print(f"  [!!]  {label:<22}: ⚠  {msg}")

    return results, all_valid


# ─────────────────────────────────────────────────────────────
#  STEP 16 — Save stage images
# ─────────────────────────────────────────────────────────────

def step16_save(stages, out="aadhaar_stages"):
    section("16 — Save Stage Images")
    os.makedirs(out, exist_ok=True)
    for name, img in stages.items():
        p = os.path.join(out, f"{name}.png")
        cv2.imwrite(p, img)
        ok(f"Saved: {p}")
    info(f"Open folder: ./{out}/")


# ─────────────────────────────────────────────────────────────
#  STEP 17 — Summary
# ─────────────────────────────────────────────────────────────

def mask_aadhaar(n):
    if not n: return "—"
    c = n.replace(" ", "")
    return f"XXXX XXXX {c[8:]}" if len(c) == 12 else n

def step17_summary(fields, verification, all_valid, qr_result=None, face_result=None):
    section("17 — Final Summary")

    W = 60  # box width

    def row(label, value):
        v = str(value or '—')[:W-22]
        print(f"  │  {label:<20}: {v:<{W-24}}│")

    def divider(title=''):
        if title:
            t = f' {title} '
            pad = W - 4 - len(t)
            print(f"  ├──{t}{'─'*pad}┤")
        else:
            print(f"  ├{'─'*(W-2)}┤")

    def header(title):
        t = f'  {title}  '
        pad = W - 2 - len(t)
        print(f"  │{t}{' '*pad}│")

    print()
    print(f"  ┌{'─'*(W-2)}┐")
    header("AADHAAR CARD — COMPLETE EXTRACTION RESULT")
    divider('IDENTITY')
    row("Name (English)",    fields.get("name"))
    row("Name (Hindi)",      fields.get("name_hindi"))
    row("Date of Birth",     fields.get("dob") or fields.get("year_of_birth"))
    row("Gender",            fields.get("gender"))
    row("Father/Husband",    fields.get("father_husband_name"))

    divider('DOCUMENT NUMBERS')
    row("Aadhaar Number",    mask_aadhaar(fields.get("aadhaar_number")))
    row("Virtual ID (VID)",  fields.get("vid"))
    row("Enrollment No",     fields.get("enrollment_number"))

    divider('CONTACT')
    row("Mobile",            fields.get("mobile"))
    row("Email",             fields.get("email"))

    divider('ADDRESS')
    row("House/Door",        fields.get("address_house"))
    row("Street",            fields.get("address_street"))
    row("Landmark",          fields.get("address_landmark"))
    row("Locality",          fields.get("address_locality"))
    row("Sub-District",      fields.get("address_subdistrict"))
    row("District",          fields.get("address_district"))
    row("State",             fields.get("address_state"))
    row("PIN Code",          fields.get("address_pin"))

    divider('META')
    row("Issue Date",        fields.get("issue_date"))
    row("Card Type",         fields.get("card_type"))
    row("Issued By",         "UIDAI")

    divider('OCR FIELD VERIFICATION')
    for label, res in verification.items():
        sym = "✓" if res["valid"] is True else ("⚠" if res["valid"] is None else "✗")
        msg = str(res['msg'])[:W-26]
        print(f"  │  {sym} {label:<19}: {msg:<{W-25}}│")

    # ── QR Trust Score ────────────────────────────────────────
    if qr_result:
        divider('QR CODE VERIFICATION')
        score   = qr_result.get('trust_score', 0)
        verdict = qr_result.get('verdict', 'UNDETERMINED')
        bar_len = int(score / 5)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        ts_line = f"{score:>3}/100  [{bar}]"
        print(f"  │  {'Trust Score':<20}: {ts_line:<{W-24}}│")
        print(f"  │  {'QR Verdict':<20}: {str(verdict):<{W-24}}│")

        for fname, fcheck in qr_result.get('field_checks', {}).items():
            m = fcheck.get('match')
            sym2 = "✓" if m is True else ("⚠" if m is None else "✗")
            label2 = f"QR {fname.replace('_',' ').title()}"
            note   = str(fcheck.get('note',''))[:W-26]
            print(f"  │  {sym2} {label2:<19}: {note:<{W-25}}│")

        signals = qr_result.get('fraud_signals', [])
        if signals:
            divider(f'⚠ FRAUD SIGNALS ({len(signals)})')
            for sig in signals[:4]:
                s = str(sig)[:W-6]
                print(f"  │  • {s:<{W-6}}│")

    # ── Face AI Results ───────────────────────────────────────
    if face_result:
        divider('FACE AI')
        row("Extraction Method",  face_result.get('extraction_method', '—'))
        row("Face Quality",
            f"{face_result.get('quality_score',0)}/100  "
            f"({face_result.get('quality_verdict','—')})")
        liveness = face_result.get('liveness_score')
        if liveness is not None:
            real_str = "Likely Real" if face_result.get('likely_real') else "Suspicious"
            row("Liveness Hint",  f"{liveness:.0f}/100  ({real_str})")

        match_res = face_result.get('match_result')
        if match_res:
            row("Face Match Engine", match_res.get('engine', '—'))
            row("Match Score",
                f"{match_res.get('match_score',0):.1f}/100")
            row("Match Verdict",  match_res.get('verdict', '—'))
            note = str(match_res.get('note',''))[:W-24]
            print(f"  │  {'Match Note':<20}: {note:<{W-24}}│")
        elif face_result.get('selfie_provided'):
            row("Face Match",  "Failed — see logs")
        else:
            row("Face Match",  "No selfie provided")

    divider()
    status = "ALL FIELDS VERIFIED ✓" if all_valid else "REVIEW REQUIRED ⚠"
    print(f"  │  Overall OCR Status : {status:<{W-25}}│")
    print(f"  └{'─'*(W-2)}┘")
    print()
