# field_extraction.py
# Step 14: Smart field extraction — all individual extractors + step14_extract()
# Extractors: name, name_hindi, dob, dob_tuples, gender, aadhaar_number,
#             vid, enrollment_number, mobile, email, father_husband_name,
#             address_structured, card_type, year_of_birth, issue_date
# ─────────────────────────────────────────────────────────────
# No code changed — same functions, split into own file.

import re

from utils import section, ok, info, warn, err
from utils import fix_digit_string

# ─────────────────────────────────────────────────────────────
#  STEP 14 — Smart field extraction
#            Uses fuzzy/flexible regex patterns to handle
#            noisy Tesseract output with | chars, quotes, etc.
# ─────────────────────────────────────────────────────────────

def clean_line(line):
    """Strip common Tesseract noise chars from a line."""
    # Remove leading/trailing: | ' ` ~ . - _ spaces
    line = re.sub(r"^[\s|'`~.\-_\\/(]+", "", line)
    line = re.sub(r"[\s|'`~.\-_\\/)]+$", "", line)
    return line.strip()

def extract_aadhaar_number(text):
    """
    Aadhaar = 12 digits in groups: XXXX XXXX XXXX
    First digit must be 2-9 (never 0 or 1)
    Tesseract may add noise chars between digit groups.
    """
    # Try strict: 4-4-4 with spaces
    m = re.search(r'\b([2-9]\d{3})\s+(\d{4})\s+(\d{4})\b', text)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}"

    # Try with any separator between groups (Tesseract noise)
    m = re.search(r'([2-9]\d{3})[^0-9]{0,3}(\d{4})[^0-9]{0,3}(\d{4})', text)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}"

    # Try contiguous 12 digits
    m = re.search(r'\b([2-9]\d{11})\b', text)
    if m:
        d = m.group(1)
        return f"{d[0:4]} {d[4:8]} {d[8:12]}"

    return None

def extract_dob(text):
    """
    Extract DOB from one OCR pass text.
    Returns list of ALL plausible (day, month, year) tuples found —
    not just the first one — so the voter gets maximum input.

    Handles every Tesseract misread pattern seen so far:
      Clean  : "DOB: 10/11/2005"
      Noisy  : "DOR 10 12006"       (space instead of /, year off by 1)
      Noisy  : "DOR 10/1 172008"    (month split: 1 and 7→1, year 2008)
      Noisy  : "DOB: lor Waid"      (pure garbage after DOB keyword)
      Noisy  : "s DOR 10 12006"     (prefix noise)
    """
    results = []

    def try_add(d_s, m_s, y_s):
        """Fix digits and add if valid."""
        try:
            d  = int(fix_digit_string(str(d_s).strip()))
            m  = int(fix_digit_string(str(m_s).strip()))
            y  = int(fix_digit_string(str(y_s).strip()))
            if 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2025:
                results.append((d, m, y))
            elif 1 <= m <= 31 and 1 <= d <= 12 and 1900 <= y <= 2025:
                # Try swapped
                results.append((m, d, y))
        except (ValueError, TypeError):
            pass

    # ── Pattern 1: clean dd/mm/yyyy or dd-mm-yyyy ────────────────
    for m in re.finditer(r'(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})', text):
        try_add(m.group(1), m.group(2), m.group(3))

    # ── Pattern 2: DOB keyword + dd/mm/yyyy ─────────────────────
    m = re.search(
        r'(?:DOB|DOR|D0B|D08|Date\s*of\s*Birth)[:\s]*'
        r'(\d{1,2})[/.\- ]{0,2}(\d{1,2})[/.\- ]{0,2}(\d{4})',
        text, re.IGNORECASE)
    if m:
        try_add(m.group(1), m.group(2), m.group(3))

    # ── Pattern 3: "DOR 10 12006" — month+year merged ───────────
    # "12006" = month "1" + year "2006" merged together
    m = re.search(
        r'(?:DOB|DOR|D0B|D08)[:\s]+(\d{1,2})\s+(\d)((?:19|20)\d{2})',
        text, re.IGNORECASE)
    if m:
        try_add(m.group(1), m.group(2), m.group(3))

    # ── Pattern 4: "DOR 10/1 172008" — year has extra digit ─────
    # "172008" = noise "1" + year "72008"? No — "1 72008" means
    # month=1, then "7" is noise, year=2008. But actual is 11/2005.
    # Better: extract ALL digit clusters near DOB keyword
    m = re.search(
        r'(?:DOB|DOR|D0B|D08)[:\s]*([\d/ .\-]{4,20})',
        text, re.IGNORECASE)
    if m:
        raw = m.group(1)
        # Pull out all digit clusters
        clusters = re.findall(r'\d+', raw)
        # Fix each cluster's digits
        clusters = [fix_digit_string(c) for c in clusters]
        # Try all reasonable interpretations
        if len(clusters) >= 3:
            try_add(clusters[0], clusters[1], clusters[2])
        if len(clusters) >= 2:
            # Maybe month+year merged: e.g. clusters = ['10', '12006']
            if len(clusters[1]) == 5:
                try_add(clusters[0], clusters[1][0], clusters[1][1:])
            # Or: clusters = ['10', '1', '2006'] already caught above
        if len(clusters) == 2 and len(clusters[1]) == 6:
            # e.g. ['10', '112005'] = day=10, month=11, year=2005
            try_add(clusters[0], clusters[1][:2], clusters[1][2:])

    # Return best single string (used by old code path for display)
    if results:
        # Pick most plausible: prefer valid month <= 12
        valid = [(d,m,y) for d,m,y in results if m <= 12]
        if valid:
            d, m, y = valid[0]
            return f"{d:02d}/{m:02d}/{y}"
        d, m, y = results[0]
        return f"{d:02d}/{m:02d}/{y}"
    return None





def extract_gender(text):
    """
    Gender on Aadhaar: Male / Female / Transgender
    Tesseract misreads seen in real output:
      "Male" → "Miso", "oso", "Wiso", "Mate", "Male", "Msle"
    Strategy: exact match first, then fuzzy phonetic patterns.
    """
    # ── Exact match ──────────────────────────────────────────────────
    m = re.search(r'\b(Male|Female|Transgender|MALE|FEMALE|TRANSGENDER)\b', text)
    if m:
        return m.group(1).capitalize()

    # ── Line-by-line fuzzy match ─────────────────────────────────────
    # "Male" is always on its own short line on Aadhaar cards
    for raw_line in text.splitlines():
        line = clean_line(raw_line).strip()

        # Short standalone line (gender is never more than 12 chars)
        if not (2 <= len(line) <= 12):
            continue

        line_up = line.upper()

        # Male variants: Mate, Msle, Miso, oso, M@le, Mal, Male
        if re.match(r'^M[a-zA-Z@]{1,3}[le]?$', line, re.IGNORECASE):
            return "Male"
        if line_up in ("MALE", "MATE", "MSLE", "MISO", "OSO", "WISO",
                       "MAL", "MALI", "MAIE", "M0LE", "MOLE"):
            return "Male"

        # Female variants
        if re.match(r'^Fe?m[a-z@]{0,4}le?$', line, re.IGNORECASE):
            return "Female"
        if line_up in ("FEMALE", "FEMLE", "FEMAL", "FEMAIE"):
            return "Female"

    # ── Context search: word near DOB line ───────────────────────────
    # On Aadhaar, Gender appears RIGHT AFTER the DOB line
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.search(r'\b(?:DOB|DOR|D0B)\b', line, re.IGNORECASE):
            # Check the next 1-3 lines for gender
            for j in range(i+1, min(i+4, len(lines))):
                nxt = clean_line(lines[j]).strip()
                if 2 <= len(nxt) <= 12:
                    up = nxt.upper()
                    if up.startswith("M"):
                        return "Male"
                    if up.startswith("F"):
                        return "Female"

    return None

def _is_hindi_line(line):
    """
    Returns True if the line contains Devanagari (Hindi) characters.
    Used to skip the Hindi transliteration line that Aadhaar cards
    print directly below the English name — Tesseract sometimes
    partially reads it as garbled English and picks it as a name.
    """
    return bool(re.search(r'[\u0900-\u097F]', line))


def extract_name(text):
    """
    IMPROVED NAME EXTRACTION — v4

    Three-strategy approach (tried in order):

    Strategy 1 — Positional anchor (most reliable):
      On every Aadhaar card the layout is fixed:
        Line 1: "Government of India"  (or Govt/GOVT OF INDIA)
        Line 2: (sometimes blank / logo noise)
        Line 3: HOLDER'S NAME  ← what we want
        Line 4: Hindi transliteration of name (skip this)
        Line 5: DOB: dd/mm/yyyy
      We find the "Government of India" anchor line and return
      the first valid English-only line that follows it and
      comes before the DOB line.

    Strategy 2 — Name-label anchor:
      Some scans have explicit labels like "Name:" or "नाम:"
      We extract the value from those labels directly.

    Strategy 3 — Scoring fallback (same as v3 but stricter):
      Used only when both anchors fail. Scores every candidate
      line and returns the best one. Now skips Hindi lines and
      has a tighter blacklist.
    """
    BLACKLIST = {
        'GOVERNMENT', 'INDIA', 'AADHAAR', 'AUTHORITY', 'UNIQUE',
        'IDENTIFICATION', 'GOVT', 'BHARAT', 'SARKAR', 'DOB', 'DATE',
        'BIRTH', 'MALE', 'FEMALE', 'ISSUE', 'VALID', 'DOWNLOAD',
        'ENROLLMENT', 'ADDRESS', 'MOBILE', 'PHONE', 'YEAR', 'OF',
        'MERA', 'PEHCHAAN', 'ENROLMENT', 'NUMBER', 'VID', 'UID',
        'HELP', 'UIDAI', 'RESIDENT', 'CARD', 'IDENTITY', 'THE',
    }

    lines = text.splitlines()

    def is_valid_name_line(line):
        """Check whether a cleaned line could be a person's name."""
        line = clean_line(line)
        if not (5 <= len(line) <= 55):          # min 5 chars — rules out 'LEE', 'EE' etc
            return False
        # Must be only English letters, spaces, dots, hyphens, apostrophes
        if not re.match(r"^[A-Za-z][A-Za-z\s.\-']+$", line):
            return False
        # Skip Hindi lines (Devanagari Unicode block)
        if _is_hindi_line(line):
            return False
        words = line.split()
        if not (2 <= len(words) <= 6):          # must have at least 2 words
            return False
        upper_words = {w.upper() for w in words}
        # Reject if ANY word is blacklisted
        if upper_words & BLACKLIST:
            return False
        # Reject if any word is fewer than 2 chars (noise like 'a', '-', '/')
        if any(len(w) < 2 for w in words):
            return False
        # Prefer longer words — genuine names rarely have words < 2 chars
        return True

    # ── Strategy 1: Positional anchor ────────────────────────
    GOI_PATTERN = re.compile(
        r'(government\s+of\s+india|govt\.?\s+of\s+india|भारत\s+सरकार)',
        re.IGNORECASE
    )
    DOB_PATTERN = re.compile(
        r'\b(DOB|DOR|D0B|Date\s*of\s*Birth)\b', re.IGNORECASE
    )

    goi_index = None
    dob_index = None

    for i, line in enumerate(lines):
        if goi_index is None and GOI_PATTERN.search(line):
            goi_index = i
        if dob_index is None and DOB_PATTERN.search(line):
            dob_index = i

    if goi_index is not None:
        # Search in the window between GOI header and DOB line
        # Use a wider window (12 lines) to handle noisy multi-pass output
        search_end = dob_index if (dob_index and dob_index > goi_index) else goi_index + 12
        for line in lines[goi_index + 1 : search_end]:
            cl = clean_line(line)
            if _is_hindi_line(cl):
                continue  # skip Devanagari line
            if is_valid_name_line(cl):
                return cl.upper() if cl == cl.upper() else cl

    # ── Strategy 1B: search near EVERY GOI occurrence ────────
    # Tesseract multi-pass output contains GOI multiple times;
    # the name appears after a different occurrence than the first.
    all_goi_indices = [i for i, l in enumerate(lines) if GOI_PATTERN.search(l)]
    for gi in all_goi_indices:
        end_i = gi + 12
        for line in lines[gi + 1 : end_i]:
            cl = clean_line(line)
            if _is_hindi_line(cl):
                continue
            if is_valid_name_line(cl):
                return cl.upper() if cl == cl.upper() else cl

    # ── Strategy 2: Name label anchor ────────────────────────
    NAME_LABEL = re.compile(
        r'^(?:Name|Full\s*Name|नाम)\s*[:\-]\s*(.+)$',
        re.IGNORECASE
    )
    for line in lines:
        m = NAME_LABEL.match(line.strip())
        if m:
            candidate = clean_line(m.group(1))
            if is_valid_name_line(candidate):
                return candidate

    # ── Strategy 3: Scoring fallback ─────────────────────────
    candidates = []
    for raw_line in lines:
        line = clean_line(raw_line)
        if _is_hindi_line(line):
            continue
        if not is_valid_name_line(line):
            continue

        words = line.split()
        score = 0
        if line == line.upper():            score += 10  # ALL CAPS like Aadhaar prints
        if 2 <= len(words) <= 4:            score += 8   # typical name length
        if len(words) == 1:                 score += 2   # single-word name (rare but valid)
        if all(len(w) >= 3 for w in words): score += 3   # no spurious single-chars
        if len(words) >= 3:                 score += 5   # three-part Indian name

        # Boost if line appears between GOI and DOB positionally
        try:
            li = lines.index(raw_line)
            if goi_index is not None and goi_index < li:
                score += 4
            if dob_index is not None and li < dob_index:
                score += 4
        except ValueError:
            pass

        candidates.append((score, line))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]

def extract_issue_date(text):
    m = re.search(r'Issue\s*Date[:\s]+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})', text, re.IGNORECASE)
    if m:
        return m.group(1)
    # Sometimes on the side: "Issue Date: 13/12/2011"
    m = re.search(r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})', text)
    # Only return if it's before the main DOB (issue dates are usually earlier)
    return None

def extract_mobile(text):
    m = re.search(r'\b([6-9]\d{9})\b', text)
    return m.group(1) if m else None


# ─────────────────────────────────────────────────────────────
#  NEW FIELD EXTRACTORS — v5
# ─────────────────────────────────────────────────────────────

def extract_name_hindi(text):
    """
    Extract the Hindi (Devanagari) name from OCR text.
    On Aadhaar cards the Hindi name is printed directly below
    the English name.  Tesseract with hin+eng lang pack reads it.

    Returns the first Devanagari-dominant line that looks like
    a name (2–5 words, no digits, not a header phrase).
    """
    HINDI_BLACKLIST_FRAGMENTS = [
        'भारत', 'सरकार', 'विशिष्ट', 'पहचान', 'प्राधिकरण',
        'आधार', 'मेरा', 'पहचान', 'नामांकन',
    ]
    for raw_line in text.splitlines():
        line = raw_line.strip()
        # Must contain Devanagari
        if not re.search(r'[\u0900-\u097F]{3,}', line):
            continue
        # Must not be a header/institution phrase
        if any(frag in line for frag in HINDI_BLACKLIST_FRAGMENTS):
            continue
        # No digits
        if re.search(r'\d', line):
            continue
        words = line.split()
        if 1 <= len(words) <= 5:
            return line.strip()
    return None


def extract_year_of_birth(text):
    """
    Some Aadhaar cards print only Year of Birth (YYYY) instead of
    full DOB.  Pattern: 'Year of Birth: 1985'  or  'YOB: 1985'
    """
    m = re.search(
        r'(?:Year\s*of\s*Birth|YOB)[:\s]+(\d{4})',
        text, re.IGNORECASE
    )
    if m:
        y = int(m.group(1))
        if 1900 <= y <= 2025:
            return str(y)
    return None


def extract_vid(text):
    """
    Virtual ID (VID) — a 16-digit temporary number UIDAI introduced
    in 2018. Printed on newer Aadhaar letters.
    Format: XXXX XXXX XXXX XXXX  (16 digits, first digit 2–9)
    Distinguished from Aadhaar number by being 16 digits.
    """
    # Strict: 4-4-4-4 with spaces
    m = re.search(r'\b([2-9]\d{3})\s+(\d{4})\s+(\d{4})\s+(\d{4})\b', text)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}"
    # Contiguous 16 digits
    m = re.search(r'\b([2-9]\d{15})\b', text)
    if m:
        d = m.group(1)
        return f"{d[0:4]} {d[4:8]} {d[8:12]} {d[12:16]}"
    return None


def extract_enrollment_number(text):
    """
    Enrollment Number (EID) — printed on old Aadhaar letters.
    Format: XXXX/XXXXX/XXXXX  or  14-digit number
    Example: 1234/12345/12345   or   12341234512345
    """
    m = re.search(r'\b(\d{4}[/\s]\d{5}[/\s]\d{5})\b', text)
    if m:
        return m.group(1).replace(' ', '/')
    m = re.search(r'\bEID[:\s]*(\d{14})\b', text, re.IGNORECASE)
    if m:
        d = m.group(1)
        return f"{d[0:4]}/{d[4:9]}/{d[9:14]}"
    return None


def extract_email(text):
    """Extract user's email address if present. Skips UIDAI's own printed email."""
    UIDAI_EMAILS = {'help@uidai.gov.in', 'support@uidai.gov.in', 'info@uidai.gov.in'}
    m = re.search(
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
        text
    )
    if m:
        email = m.group(0).lower()
        if email in UIDAI_EMAILS:
            return None   # This is UIDAI's own contact email, not the cardholder's
        return email
    return None


def extract_father_husband_name(text):
    """
    Extract C/O (Care Of) / Father / Husband name.
    Aadhaar address block starts with:
      'C/O: FATHER NAME'  or  'S/O FATHER'  or  'W/O HUSBAND'
      or  'D/O FATHER'  or  'F/O FATHER'
    """
    # Explicit label
    m = re.search(
        r'\b(?:C[/\\]O|S[/\\]O|W[/\\]O|D[/\\]O|F[/\\]O|'
        r'Father|Husband|Guardian|Son of|Wife of|Daughter of)'
        r'[\s:,\.]+([A-Za-z][A-Za-z\s.\-\']{2,40})',
        text, re.IGNORECASE
    )
    if m:
        candidate = clean_line(m.group(1))
        if 2 <= len(candidate.split()) <= 5:
            return candidate.strip()

    # Pattern: line starting with S/O or C/O
    for line in text.splitlines():
        stripped = line.strip()
        m2 = re.match(
            r'^(?:C/O|S/O|W/O|D/O|F/O)[:\s]+(.+)$',
            stripped, re.IGNORECASE
        )
        if m2:
            candidate = clean_line(m2.group(1))
            words = candidate.split()
            if 1 <= len(words) <= 5 and re.match(r'^[A-Za-z\s.\-\']+$', candidate):
                return candidate.strip()
    return None


def extract_address_structured(text):
    """
    Extract full address from Aadhaar OCR text and return it as both
    a raw string and a structured dict with individual components.

    Aadhaar address layout (after the photo strip):
      C/O: FATHER NAME
      House No / Door No, Street / Lane / Road,
      Landmark (near / opp),
      Village / Town / City,
      Sub-District, District,
      State – PIN CODE

    Strategy:
      1. Find the address block start: line after DOB/Gender
         that contains C/O, House, or a known address keyword.
      2. Collect consecutive lines until we hit Aadhaar number
         or end of content.
      3. Parse PIN code, state, district, and other components
         from the collected block.

    Returns: dict with keys:
        raw, house, street, landmark, locality, subdistrict,
        district, state, pin_code, full_formatted
    """
    addr = {
        'raw':          None,
        'house':        None,
        'street':       None,
        'landmark':     None,
        'locality':     None,
        'subdistrict':  None,
        'district':     None,
        'state':        None,
        'pin_code':     None,
        'full_formatted': None,
    }

    lines = text.splitlines()

    # ── Find address block start ──────────────────────────────
    ADDR_START_PATTERNS = [
        re.compile(r'\b(C[/\\]O|S[/\\]O|W[/\\]O|House\s*No|H\.?No|Door\s*No)',
                   re.IGNORECASE),
        re.compile(r'\b(Near|Opp|Behind|Beside|Next\s*to)\b', re.IGNORECASE),
        re.compile(r'\b(Village|Vill\.?|Post|P\.?O\.?|Taluka|Tehsil)\b',
                   re.IGNORECASE),
    ]
    ADDR_END_PATTERNS = [
        re.compile(r'\b\d{4}\s+\d{4}\s+\d{4}\b'),   # Aadhaar number
        re.compile(r'\bVID\b', re.IGNORECASE),
        re.compile(r'Download\s*Date', re.IGNORECASE),
        re.compile(r'Issue\s*Date', re.IGNORECASE),
    ]

    start_idx = None
    # Also anchor after gender/DOB line
    gender_dob_idx = None
    for i, line in enumerate(lines):
        if re.search(r'\b(Male|Female|Transgender|DOB|DOR)\b', line, re.IGNORECASE):
            gender_dob_idx = i
    if gender_dob_idx is not None:
        search_from = gender_dob_idx + 1
    else:
        search_from = 0

    for i in range(search_from, len(lines)):
        for pat in ADDR_START_PATTERNS:
            if pat.search(lines[i]):
                start_idx = i
                break
        if start_idx is not None:
            break

    if start_idx is None:
        # Fallback: grab last 6–10 lines before Aadhaar number line
        for i, line in enumerate(lines):
            if re.search(r'\b[2-9]\d{3}\s+\d{4}\s+\d{4}\b', line):
                start_idx = max(0, i - 8)
                break

    if start_idx is None:
        return addr

    # ── Collect address lines ─────────────────────────────────
    addr_lines = []
    for i in range(start_idx, min(start_idx + 12, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        # Stop if we hit Aadhaar number or other terminal patterns
        stop = False
        for pat in ADDR_END_PATTERNS:
            if pat.search(line):
                stop = True
                break
        if stop:
            break
        # Skip very short noise lines
        if len(line) < 3:
            continue
        # Strip leading OCR noise tokens (1-3 char symbols/codes before real text)
        # e.g. "BS poonam nivas" → "poonam nivas"
        #      "§ near kamla"    → "near kamla"
        line = re.sub(r'^[^A-Za-z0-9]{0,3}[A-Z]{1,2}\s+', '', line).strip()
        line = re.sub(r'^[^A-Za-z0-9]+', '', line).strip()
        if len(line) < 3:
            continue
        addr_lines.append(line)

    if not addr_lines:
        return addr

    raw_addr = ' '.join(addr_lines)
    # Remove trailing OCR noise fragments like "eae eas eee"
    raw_addr = re.sub(r'\s+[a-z]{1,3}(\s+[a-z]{1,3}){2,}$', '', raw_addr).strip()
    addr['raw'] = raw_addr

    # ── Extract PIN code ──────────────────────────────────────
    pin_m = re.search(r'\b([1-9]\d{5})\b', raw_addr)
    if pin_m:
        addr['pin_code'] = pin_m.group(1)

    # ── Extract State ─────────────────────────────────────────
    INDIAN_STATES = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar',
        'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh',
        'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
        'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
        'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
        'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
        'Andaman and Nicobar', 'Chandigarh', 'Delhi', 'Jammu and Kashmir',
        'Ladakh', 'Lakshadweep', 'Puducherry', 'Dadra', 'Daman',
    ]
    for state in INDIAN_STATES:
        if re.search(re.escape(state), raw_addr, re.IGNORECASE):
            addr['state'] = state
            break

    # ── Extract C/O (house guardian name) ────────────────────
    co_m = re.match(r'C[/\\]O[:\s]+([A-Za-z\s.\-\']+)', raw_addr, re.IGNORECASE)
    if co_m:
        addr['house'] = clean_line(co_m.group(1)).strip()

    # ── Extract House/Door number ─────────────────────────────
    house_m = re.search(
        r'(?:House\s*No|H\.?No|Door\s*No|Plot\s*No|Flat\s*No|'
        r'#|Room\s*No)[\.:\s]*([A-Za-z0-9/\-]+)',
        raw_addr, re.IGNORECASE
    )
    if house_m and not addr['house']:
        addr['house'] = house_m.group(1).strip()

    # ── Extract Landmark ──────────────────────────────────────
    lm_m = re.search(
        r'(?:Near|Opp\.?|Opposite|Behind|Beside|Next\s*to|'
        r'Landmark)[\.:\s]+([A-Za-z0-9\s,.\-]{3,40})',
        raw_addr, re.IGNORECASE
    )
    if lm_m:
        addr['landmark'] = lm_m.group(1).strip().rstrip(',')

    # ── Extract Village/Town/City ─────────────────────────────
    loc_m = re.search(
        r'(?:Village|Vill\.?|Town|City|Locality|Area|Nagar|'
        r'Mohalla|Colony|Ward)[\.:\s]+([A-Za-z\s.\-]{2,30})',
        raw_addr, re.IGNORECASE
    )
    if loc_m:
        addr['locality'] = loc_m.group(1).strip().rstrip(',')

    # ── Extract District ──────────────────────────────────────
    dist_m = re.search(
        r'(?:Dist\.?|District)[\.:\s]+([A-Za-z\s.\-]{2,25})',
        raw_addr, re.IGNORECASE
    )
    if dist_m:
        addr['district'] = dist_m.group(1).strip().rstrip(',')

    # ── Extract Sub-District / Taluka / Tehsil ────────────────
    sub_m = re.search(
        r'(?:Sub[\s\-]?Dist\.?|Taluka|Tehsil|Mandal|Block)[\.:\s]+'
        r'([A-Za-z\s.\-]{2,25})',
        raw_addr, re.IGNORECASE
    )
    if sub_m:
        addr['subdistrict'] = sub_m.group(1).strip().rstrip(',')

    # ── Build formatted address ───────────────────────────────
    parts = [p for p in [
        addr.get('house'), addr.get('street'),
        addr.get('landmark'), addr.get('locality'),
        addr.get('subdistrict'), addr.get('district'),
        addr.get('state'), addr.get('pin_code'),
    ] if p]
    addr['full_formatted'] = ', '.join(parts) if parts else raw_addr

    return addr


def extract_card_type(text):
    """
    Determine whether this is Front or Back of the Aadhaar card,
    or a single-page e-Aadhaar (PDF download).

    Front: has name, DOB, gender, photo
    Back:  has address, barcode, usually no DOB
    e-Aadhaar: has both sides merged into one image
    """
    has_dob    = bool(re.search(r'\b(DOB|Date\s*of\s*Birth)\b', text, re.IGNORECASE))
    has_addr   = bool(re.search(r'\b(District|Dist|State|PIN|Pincode)\b', text, re.IGNORECASE))
    has_name   = bool(re.search(r'Government\s*of\s*India', text, re.IGNORECASE))

    if has_dob and has_addr:
        return "e-Aadhaar / Combined (both sides)"
    elif has_dob and has_name:
        return "Front Side"
    elif has_addr and not has_dob:
        return "Back Side"
    else:
        return "Unknown"

def step14_extract(combined_text):
    section("14 — Smart Field Extraction (All Fields)")

    info("Extracting every field available on an Aadhaar card...")
    print()

    # ── Structured address extraction ────────────────────────
    addr = extract_address_structured(combined_text)

    fields = {
        # Identity
        "name":              extract_name(combined_text),
        "name_hindi":        extract_name_hindi(combined_text),
        "dob":               extract_dob(combined_text),
        "year_of_birth":     extract_year_of_birth(combined_text),
        "gender":            extract_gender(combined_text),
        # Document numbers
        "aadhaar_number":    extract_aadhaar_number(combined_text),
        "vid":               extract_vid(combined_text),
        "enrollment_number": extract_enrollment_number(combined_text),
        # Relations
        "father_husband_name": extract_father_husband_name(combined_text),
        # Contact
        "mobile":            extract_mobile(combined_text),
        "email":             extract_email(combined_text),
        # Address (structured)
        "address_raw":       addr.get('raw'),
        "address_house":     addr.get('house'),
        "address_street":    addr.get('street'),
        "address_landmark":  addr.get('landmark'),
        "address_locality":  addr.get('locality'),
        "address_subdistrict": addr.get('subdistrict'),
        "address_district":  addr.get('district'),
        "address_state":     addr.get('state'),
        "address_pin":       addr.get('pin_code'),
        # Meta
        "issue_date":        extract_issue_date(combined_text),
        "card_type":         extract_card_type(combined_text),
        "issued_by":         "Unique Identification Authority of India",
    }

    # Print results with clear markers
    SKIP_PRINT = {'issued_by'}
    for k, v in fields.items():
        if k in SKIP_PRINT:
            continue
        marker = "  [OK] " if v else "  [..] "
        display = str(v)[:60] if v else "not found"
        print(f"{marker}  {k:<28}: {display}")

    return fields

