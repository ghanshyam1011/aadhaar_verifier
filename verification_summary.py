# verification_summary.py
# Step 15: Field verification (format checks)
# Step 16: Save stage images to disk
# Step 17: Final summary — 4-column verification table
#
# TABLE FORMAT:
#   Field Name | OCR (from image) | QR (from code) | Conclusion
#
# MINIMUM FIELDS FOR VALID AADHAAR:
#   MANDATORY (all 4 must pass):
#     1. Name          — OCR extracted + QR exact/fuzzy match
#     2. Date of Birth — OCR extracted + QR exact match
#     3. Gender        — OCR extracted + QR match
#     4. Aadhaar No.   — OCR 12-digit valid format
#                        (QR match is bonus — Secure QR omits it)
#
#   STRONGLY RECOMMENDED (at least 2 of 3):
#     5. PIN Code      — OCR + QR exact match
#     6. District      — OCR + QR match
#     7. State         — OCR + QR match
#
#   VERDICT RULES:
#     VERIFIED       : All 4 mandatory pass + QR trust >= 80
#     LIKELY GENUINE : All 4 mandatory pass + QR trust 60-79
#     REVIEW NEEDED  : 3 of 4 mandatory pass OR QR trust 40-59
#     REJECTED       : < 3 mandatory OR QR trust < 40
# ─────────────────────────────────────────────────────────────

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
        if g.upper() in ["MALE", "FEMALE", "TRANSGENDER"]: return True, "Valid ✓"
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
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _mask_aadhaar(n):
    if not n: return "—"
    c = n.replace(" ", "")
    return f"XXXX XXXX {c[8:]}" if len(c) == 12 else n


def _trunc(val, max_len):
    """Truncate a value to max_len chars, pad with spaces."""
    s = str(val) if val else "—"
    if len(s) > max_len:
        s = s[:max_len - 1] + "…"
    return s


def _conclusion(ocr_val, qr_val, qr_match, mandatory=False):
    """
    Return (symbol, text) for the Conclusion column.
    ocr_val  : value from OCR (may be None)
    qr_val   : value from QR  (may be None)
    qr_match : True / False / None (None = QR field not available)
    mandatory: bool — is this a mandatory field?
    """
    no_ocr = not ocr_val or str(ocr_val).strip() in ('', '—', 'None')
    no_qr  = not qr_val  or str(qr_val).strip()  in ('', '—', 'None')

    if no_ocr and no_qr:
        return "✗", "Not found in OCR or QR"
    if no_ocr:
        return "⚠", f"Only in QR: {str(qr_val)[:25]}"
    if no_qr or qr_match is None:
        # OCR found but no QR to compare — valid only for non-mandatory or Aadhaar#
        if mandatory:
            return "⚠", "OCR only — QR unavailable"
        return "✓", "OCR extracted (no QR check)"
    if qr_match is True:
        return "✓", "OCR matches QR ✓"
    # qr_match is False
    return "✗", f"OCR ≠ QR  (OCR: {str(ocr_val)[:15]})"


# ─────────────────────────────────────────────────────────────
#  MINIMUM FIELD VERDICT ENGINE
# ─────────────────────────────────────────────────────────────

def _compute_final_verdict(fields, qr_result, verification, face_result=None):
    """
    Compute the overall Aadhaar validity verdict.

    MINIMUM FIELDS REQUIRED:
      Mandatory (all 4 must pass for a proper verification):
        1. Name          — OCR valid format
        2. Date of Birth — OCR valid format
        3. Gender        — OCR valid format
        4. Aadhaar No.   — OCR valid 12-digit format

      QR cross-check adds confidence:
        Name match    → +1 qr_pass
        DOB match     → +1 qr_pass
        Gender match  → +1 qr_pass
        PIN match     → +1 qr_pass
        District match→ +1 qr_pass
        State match   → +1 qr_pass

    VERDICT TIERS:
      VERIFIED        : 4/4 mandatory + QR trust >= 80
      LIKELY GENUINE  : 4/4 mandatory + QR trust 60–79
                        OR 4/4 mandatory + 4+ qr_passes
      REVIEW REQUIRED : 3/4 mandatory pass
                        OR 4/4 mandatory + QR trust 40–59
      REJECTED        : < 3 mandatory pass
                        OR QR trust < 40 with mismatches
    """
    qr_trust    = qr_result.get('trust_score', 0) if qr_result else 0
    field_checks = qr_result.get('field_checks', {}) if qr_result else {}

    # Count mandatory OCR passes
    mandatory_labels = ["Name", "Date of Birth", "Gender", "Aadhaar Number"]
    mandatory_pass = sum(
        1 for lbl in mandatory_labels
        if verification.get(lbl, {}).get('valid') is True
    )

    # Count QR field matches
    qr_passes = sum(
        1 for fc in field_checks.values()
        if fc.get('match') is True
    )
    qr_total_checked = sum(
        1 for fc in field_checks.values()
        if fc.get('match') is not None
    )
    qr_mismatches = sum(
        1 for fc in field_checks.values()
        if fc.get('match') is False
    )

    # Face liveness check
    face_ok = True
    if face_result:
        liveness = face_result.get('liveness_score', 100)
        face_ok  = liveness >= 40  # must not be flagged as spoofed

    # Determine verdict
    if mandatory_pass < 3:
        verdict      = "REJECTED ✗"
        verdict_color= "red"
        reason       = (f"Only {mandatory_pass}/4 mandatory fields found. "
                        f"Minimum 4 required for any verification.")

    elif mandatory_pass == 3:
        verdict      = "REVIEW REQUIRED ⚠"
        verdict_color= "yellow"
        reason       = ("3/4 mandatory fields found. "
                        "One mandatory field missing — manual review needed.")

    elif qr_trust >= 80 and qr_mismatches == 0 and face_ok:
        verdict      = "VERIFIED ✓"
        verdict_color= "green"
        reason       = (f"All 4 mandatory OCR fields valid. "
                        f"QR trust score {qr_trust}/100 with "
                        f"{qr_passes}/{qr_total_checked} field matches.")

    elif qr_trust >= 60 or qr_passes >= 4:
        verdict      = "LIKELY GENUINE ✓"
        verdict_color= "green"
        reason       = (f"4/4 mandatory OCR fields valid. "
                        f"QR trust {qr_trust}/100 ({qr_passes} field matches).")

    elif qr_trust >= 40:
        verdict      = "REVIEW REQUIRED ⚠"
        verdict_color= "yellow"
        reason       = (f"4/4 mandatory OCR fields valid but "
                        f"QR trust only {qr_trust}/100. Manual review advised.")

    elif qr_mismatches > 0:
        verdict      = "REVIEW REQUIRED ⚠"
        verdict_color= "yellow"
        reason       = (f"4/4 mandatory OCR fields valid but "
                        f"{qr_mismatches} QR field mismatch(es) detected.")

    else:
        verdict      = "LIKELY GENUINE ✓"
        verdict_color= "green"
        reason       = (f"4/4 mandatory OCR fields valid. "
                        f"QR not available for cross-check.")

    return {
        'verdict':        verdict,
        'verdict_color':  verdict_color,
        'reason':         reason,
        'mandatory_pass': mandatory_pass,
        'qr_passes':      qr_passes,
        'qr_mismatches':  qr_mismatches,
        'qr_trust':       qr_trust,
        'face_ok':        face_ok,
    }


# ─────────────────────────────────────────────────────────────
#  STEP 17 — Summary with 4-column verification table
# ─────────────────────────────────────────────────────────────

def step17_summary(fields, verification, all_valid,
                   qr_result=None, face_result=None):
    section("17 — Final Verification Report")

    qr_fields     = (qr_result or {}).get('qr_fields', {})
    field_checks  = (qr_result or {}).get('field_checks', {})
    qr_trust      = (qr_result or {}).get('trust_score', 0)
    qr_verdict_str= (qr_result or {}).get('verdict', 'N/A')
    qr_format     = (qr_result or {}).get('qr_format', 'N/A')

    # ── Compute final verdict ─────────────────────────────────
    final = _compute_final_verdict(fields, qr_result, verification, face_result)

    # ── Column widths ─────────────────────────────────────────
    C0 = 22   # Field name
    C1 = 28   # OCR value
    C2 = 28   # QR value
    C3 = 26   # Conclusion
    TW = C0 + C1 + C2 + C3 + 7  # total width (separators)

    def hdr_row():
        print(f"  ┌{'─'*C0}┬{'─'*C1}┬{'─'*C2}┬{'─'*C3}┐")
        print(f"  │{'Field':<{C0}}│{'OCR — from image':<{C1}}│"
              f"{'QR — from code':<{C2}}│{'Conclusion':<{C3}}│")
        print(f"  ├{'─'*C0}┼{'─'*C1}┼{'─'*C2}┼{'─'*C3}┤")

    def section_row(title):
        t = f" {title} "
        pad = TW - len(t)
        print(f"  ├{t}{'─'*pad}┤")

    def data_row(field, ocr_val, qr_val, sym, conclusion, mandatory=False):
        m_mark = " *" if mandatory else "  "
        f_str  = _trunc(field + m_mark, C0)
        o_str  = _trunc(ocr_val, C1)
        q_str  = _trunc(qr_val,  C2)
        c_str  = _trunc(f"{sym} {conclusion}", C3)
        print(f"  │{f_str:<{C0}}│{o_str:<{C1}}│{q_str:<{C2}}│{c_str:<{C3}}│")

    def end_row():
        print(f"  └{'─'*C0}┴{'─'*C1}┴{'─'*C2}┴{'─'*C3}┘")

    def separator_row():
        print(f"  ├{'─'*C0}┼{'─'*C1}┼{'─'*C2}┼{'─'*C3}┤")

    print()

    # ── Table header ─────────────────────────────────────────
    hdr_row()

    # ═══════════════════════════════════════════════
    #  SECTION: IDENTITY  (mandatory fields)
    # ═══════════════════════════════════════════════
    section_row('IDENTITY — mandatory fields marked *')

    # Name
    ocr_name = fields.get('name')
    qr_name  = qr_fields.get('name')
    nm_check = field_checks.get('name', {})
    sym, con = _conclusion(ocr_name, qr_name, nm_check.get('match'), mandatory=True)
    data_row("Name", ocr_name, qr_name, sym, con, mandatory=True)

    # Date of Birth
    ocr_dob = fields.get('dob') or fields.get('year_of_birth')
    qr_dob  = qr_fields.get('dob')
    db_check = field_checks.get('dob', {})
    sym, con = _conclusion(ocr_dob, qr_dob, db_check.get('match'), mandatory=True)
    data_row("Date of Birth", ocr_dob, qr_dob, sym, con, mandatory=True)

    # Gender
    ocr_gen = fields.get('gender')
    qr_gen  = qr_fields.get('gender')
    gn_check = field_checks.get('gender', {})
    sym, con = _conclusion(ocr_gen, qr_gen, gn_check.get('match'), mandatory=True)
    data_row("Gender", ocr_gen, qr_gen, sym, con, mandatory=True)

    # Father / Husband
    ocr_fh = fields.get('father_husband_name')
    qr_fh  = qr_fields.get('care_of')
    sym, con = _conclusion(ocr_fh, qr_fh, None)
    data_row("Father / C/O", ocr_fh, qr_fh, sym, con)

    separator_row()

    # ═══════════════════════════════════════════════
    #  SECTION: DOCUMENT NUMBERS
    # ═══════════════════════════════════════════════
    section_row('DOCUMENT NUMBERS — mandatory *')

    # Aadhaar Number
    ocr_aad = fields.get('aadhaar_number')
    qr_aad  = qr_fields.get('aadhaar_number')   # None for Secure QR
    ad_check = field_checks.get('aadhaar_number', {})
    ad_match = ad_check.get('match')

    # For Secure QR, Aadhaar# is intentionally absent — show OCR only verdict
    if qr_format == 'secure_qr':
        ocr_fmt_ok = (verification.get('Aadhaar Number', {}).get('valid') is True)
        sym = "✓" if ocr_fmt_ok else "✗"
        con = "Valid 12-digit format" if ocr_fmt_ok else "Invalid format"
        qr_aad_display = "Not in Secure QR*"
    else:
        sym, con = _conclusion(ocr_aad, qr_aad, ad_match, mandatory=True)
        qr_aad_display = qr_aad

    data_row("Aadhaar Number",
             _mask_aadhaar(ocr_aad),
             qr_aad_display,
             sym, con, mandatory=True)

    # VID
    ocr_vid = fields.get('vid')
    data_row("Virtual ID (VID)", ocr_vid, "—", "✓" if ocr_vid else "—",
             "OCR extracted" if ocr_vid else "Not found")

    separator_row()

    # ═══════════════════════════════════════════════
    #  SECTION: ADDRESS (recommended 2 of 3 match)
    # ═══════════════════════════════════════════════
    section_row('ADDRESS — 2 of 3 location fields recommended')

    # PIN Code
    ocr_pin = fields.get('address_pin')
    qr_pin  = qr_fields.get('pin')
    pn_check = field_checks.get('pin', {})
    sym, con = _conclusion(ocr_pin, qr_pin, pn_check.get('match'))
    data_row("PIN Code", ocr_pin, qr_pin, sym, con)

    # District
    ocr_dist = fields.get('address_district')
    qr_dist  = qr_fields.get('district') or qr_fields.get('district2')
    dt_check = field_checks.get('district', {})
    sym, con = _conclusion(ocr_dist, qr_dist, dt_check.get('match'))
    data_row("District", ocr_dist, qr_dist, sym, con)

    # State
    ocr_state = fields.get('address_state')
    qr_state  = qr_fields.get('state')
    st_check  = field_checks.get('state', {})
    sym, con  = _conclusion(ocr_state, qr_state, st_check.get('match'))
    data_row("State", ocr_state, qr_state, sym, con)

    # House / Building
    ocr_house = fields.get('address_house')
    qr_house  = qr_fields.get('building')
    sym, con  = _conclusion(ocr_house, qr_house, None)
    data_row("House / Building", ocr_house, qr_house, sym, con)

    # Landmark
    ocr_lm = fields.get('address_landmark')
    qr_lm  = qr_fields.get('landmark')
    sym, con = _conclusion(ocr_lm, qr_lm, None)
    data_row("Landmark", ocr_lm, qr_lm, sym, con)

    # Locality
    ocr_loc = fields.get('address_locality')
    qr_loc  = qr_fields.get('locality')
    sym, con = _conclusion(ocr_loc, qr_loc, None)
    data_row("Locality", ocr_loc, qr_loc, sym, con)

    separator_row()

    # ═══════════════════════════════════════════════
    #  SECTION: CONTACT
    # ═══════════════════════════════════════════════
    section_row('CONTACT')

    ocr_mob = fields.get('mobile')
    qr_mob  = qr_fields.get('mobile_masked')
    sym, con = _conclusion(ocr_mob, qr_mob, None)
    data_row("Mobile", ocr_mob, qr_mob, sym, con)

    ocr_email = fields.get('email')
    qr_email_flag = ("Registered" if qr_fields.get('email_registered') else
                     "Not registered") if qr_fields.get('email_registered') is not None else None
    sym, con = _conclusion(ocr_email, qr_email_flag, None)
    data_row("Email", ocr_email, qr_email_flag, sym, con)

    separator_row()

    # ═══════════════════════════════════════════════
    #  SECTION: FACE AI
    # ═══════════════════════════════════════════════
    section_row('FACE AI')

    if face_result:
        q_score   = face_result.get('quality_score', 0)
        q_verdict = face_result.get('quality_verdict', '—')
        live_score = face_result.get('liveness_score')
        live_real  = face_result.get('likely_real')
        match_res  = face_result.get('match_result')

        face_sym = "✓" if q_score >= 60 else "⚠"
        data_row("Face Quality",
                 f"{q_score}/100 ({q_verdict})", "N/A",
                 face_sym,
                 "Good quality" if q_score >= 75 else "Acceptable" if q_score >= 45 else "Poor")

        if live_score is not None:
            live_sym = "✓" if live_real else "⚠"
            data_row("Liveness",
                     f"{live_score:.0f}/100", "N/A",
                     live_sym,
                     "Likely real" if live_real else "Suspicious — review")

        if match_res:
            ms = match_res.get('match_score', 0)
            mv = match_res.get('verdict', '—')
            data_row("Selfie Match",
                     f"{ms:.0f}/100", "N/A",
                     "✓" if match_res.get('match') else "✗",
                     str(mv)[:C3 - 2])
        else:
            data_row("Selfie Match", "No selfie provided", "N/A", "—", "Skipped")
    else:
        data_row("Face AI", "Not run", "N/A", "—", "Skipped")

    end_row()

    # ── QR metadata footer ────────────────────────────────────
    print()
    print(f"  QR Format      : {qr_format or 'N/A'}")
    print(f"  QR Trust Score : {qr_trust}/100")
    print(f"  QR Verdict     : {qr_verdict_str}")
    if qr_fields.get('reference_id'):
        print(f"  QR Reference   : {qr_fields['reference_id']}")
    if qr_fields.get('mobile_last4'):
        print(f"  Mobile last 4  : {qr_fields['mobile_last4']}")
    if qr_format == 'secure_qr':
        print(f"  * Aadhaar number is intentionally absent from Secure QR")
        print(f"    (UIDAI privacy design — verified via format check only)")

    # ── Minimum field summary ─────────────────────────────────
    print()
    print(f"  {'─'*TW}")
    print(f"  MINIMUM FIELD REQUIREMENTS")
    print(f"  {'─'*TW}")

    mandatory_labels = ["Name", "Date of Birth", "Gender", "Aadhaar Number"]
    for lbl in mandatory_labels:
        res   = verification.get(lbl, {})
        valid = res.get('valid')
        sym   = "✓ PASS" if valid is True else ("⚠ WARN" if valid is None else "✗ FAIL")
        print(f"  [MANDATORY]  {lbl:<22}: {sym}")

    print()
    loc_fields = [
        ("PIN Code",  field_checks.get('pin',      {}).get('match')),
        ("District",  field_checks.get('district', {}).get('match')),
        ("State",     field_checks.get('state',    {}).get('match')),
    ]
    loc_pass = sum(1 for _, m in loc_fields if m is True)
    for lbl, m in loc_fields:
        sym = "✓ PASS" if m is True else ("— SKIP" if m is None else "✗ FAIL")
        print(f"  [RECOMMENDED]{lbl:<23}: {sym}")
    print(f"  Location fields matched: {loc_pass}/3 "
          f"({'OK' if loc_pass >= 2 else 'need 2+ for strong verification'})")

    # ── FINAL VERDICT ─────────────────────────────────────────
    print()
    print(f"  {'═'*TW}")

    v = final['verdict']
    if "VERIFIED" in v and "LIKELY" not in v:
        border = "█"
    elif "LIKELY" in v or "REVIEW" not in v:
        border = "▓"
    else:
        border = "░"

    print(f"  {border*3}  FINAL VERDICT : {v}")
    print(f"  {border*3}  REASON        : {final['reason']}")
    print()
    print(f"  Mandatory fields  : {final['mandatory_pass']}/4 passed")
    print(f"  QR cross-checks   : {final['qr_passes']} matched"
          + (f", {final['qr_mismatches']} mismatched" if final['qr_mismatches'] else ""))
    print(f"  QR trust score    : {final['qr_trust']}/100")
    print()

    # Minimum fields guide
    print(f"  GUIDE — what constitutes a valid Aadhaar verification:")
    print(f"  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │ VERIFIED        : 4/4 mandatory + QR trust >= 80, 0 mismatch │")
    print(f"  │ LIKELY GENUINE  : 4/4 mandatory + QR trust 60–79             │")
    print(f"  │ REVIEW REQUIRED : 3/4 mandatory OR QR trust 40–59            │")
    print(f"  │ REJECTED        : < 3 mandatory OR active fraud signals       │")
    print(f"  └─────────────────────────────────────────────────────────────┘")
    print(f"  {'═'*TW}")
    print()