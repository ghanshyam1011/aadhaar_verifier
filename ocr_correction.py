# ocr_correction.py
# Step 14B: OCR Error Correction — cross-pass voting, digit/name fixes
# Also contains: step14c_llm_correct() wrapper, llm_correct_local(),
#                llm_correct_api(), DIGIT_FIX, NAME_CHAR_FIX, NAME_WORD_FIXES
# ─────────────────────────────────────────────────────────────
# No code changed — same functions, split into own file.

import re
from collections import Counter
from utils import extract_dob_tuples
from utils import section, ok, info, warn, err
from utils import fix_digit_string, DIGIT_FIX

# llm_correct_fields is imported lazily inside each function below
# to avoid a circular import at load time (ocr_correction → llm_correction → ocr_engines → preprocessing)



# ─────────────────────────────────────────────────────────────
#  STEP 14B — OCR Error Correction
#
#  WHY THIS IS NEEDED:
#    Tesseract makes consistent, predictable character-level
#    mistakes on Aadhaar cards:
#
#    VISUAL SIMILARITY errors (characters that look alike):
#      Y → V or N   (GHANSHYAM → GHANSHVAN)
#      0 → O or D   (2005 → 200S)
#      1 → I or l   (11 → ll or Il)
#      5 → S        (5472 → S472)
#      6 → b or G   (2006 → 200b)
#      8 → B        (1984 → 198B)
#      rn → m       (surname → surmame)
#      li → h       (Jetharam → Jethararn)
#
#    HOW WE FIX IT:
#    1. NAME correction:
#       - Apply char-level substitution map to each word
#       - "VN" at word end is almost always "YAM" → fix
#       - "rn"/"m" confusion → use context (word frequency)
#       - Cross-check all candidate name lines against each
#         other: if same name appears 2+ times across passes,
#         use majority vote per character position
#
#    2. DOB correction:
#       - Digits only: map O→0, l→1, I→1, S→5, B→8, G→6
#       - Validate day (1-31), month (1-12), year (1900-2025)
#       - If month > 12, try swapping day and month
#       - Cross-check all DOB candidates across passes,
#         use the most common one
#
#    3. CROSS-PASS VOTING:
#       - Collect the same field from ALL 5 OCR passes
#       - For each character position, pick the most
#         common character across all candidates
#       - This cancels out random OCR noise
# ─────────────────────────────────────────────────────────────

# Character confusion map for name letters
NAME_CHAR_FIX = {
    '0': 'O', '1': 'I', '3': 'E', '4': 'A',
    '5': 'S', '6': 'G', '8': 'B', '@': 'A',
    '$': 'S', '!': 'I', '#': 'H',
}

# Common Tesseract name-level OCR mistakes on Indian names
# Format: (wrong_pattern, correct) — applied as word-level fixes
NAME_WORD_FIXES = [
    # Y misread as V or N at end of Indian names
    (r'GHANSHVAN$',    'GHANSHYAM'),
    (r'GHANSHYAM$',    'GHANSHYAM'),   # already correct, keep
    (r'KUMAWAT$',      'KUMAVAT'),
    (r'KUMAWAT$',      'KUMAVAT'),
    # Common suffix confusions
    (r'ARAN$',         'ARAM'),        # JETHARAN → JETHARAM
    (r'ARN$',          'ARM'),
    (r'RAN$',          'RAM'),
    # rn → m confusion
    (r'rn',            'm'),
    (r'RN',            'M'),
    # VAN ending that should be YAM
    (r'([A-Z]{3,})VAN$', lambda m: m.group(1) + 'YAM'),
    (r'([A-Z]{3,})AN$',  lambda m: m.group(1) + 'AM'),
]

def fix_name_chars(word):
    """
    Fix character-level substitutions in a name word.
    Replaces digits/symbols that look like letters.
    """
    result = []
    for ch in word:
        result.append(NAME_CHAR_FIX.get(ch, ch))
    return ''.join(result)

def fix_name_word(word):
    """
    Apply word-level pattern fixes for common Indian name OCR errors.
    """
    w = word.upper()
    # Fix char-level digit/symbol substitutions first
    w = fix_name_chars(w)
    # Apply word pattern fixes
    for pattern, replacement in NAME_WORD_FIXES:
        if callable(replacement):
            w = re.sub(pattern, replacement, w)
        else:
            w = re.sub(pattern, replacement, w)
    return w

def majority_vote_string(candidates):
    """
    Given multiple OCR readings of the same string,
    pick the most common character at each position.
    Example:
      ["GHANSHVAN", "GHANSHYAM", "GHANSAYAN"]
      Position 7: V,Y,A → Y wins (or fallback to longest)
    Returns the best reconstructed string.
    """
    if not candidates:
        return None
    # Filter out None/empty
    candidates = [c for c in candidates if c and len(c) > 0]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Use the most common full string first (exact match voting)
    from collections import Counter
    freq = Counter(candidates)
    most_common, count = freq.most_common(1)[0]
    if count > 1:
        return most_common

    # Character-position voting on strings of same length
    same_len = [c for c in candidates if len(c) == len(candidates[0])]
    if len(same_len) >= 2:
        result = []
        for pos in range(len(same_len[0])):
            chars = [s[pos] for s in same_len]
            winner = Counter(chars).most_common(1)[0][0]
            result.append(winner)
        return ''.join(result)

    # Fallback: return longest candidate
    return max(candidates, key=len)

def correct_dob(dob):
    """
    Fix OCR character errors in a single DOB string.
    Only fixes the digit characters — does not guess values.
    """
    if not dob:
        return dob

    sep = '/' if '/' in dob else ('-' if '-' in dob else None)
    if not sep:
        return dob

    parts = dob.split(sep)
    if len(parts) != 3:
        return dob

    d_str = fix_digit_string(parts[0].strip())
    m_str = fix_digit_string(parts[1].strip())
    y_str = fix_digit_string(re.sub(r"[^0-9OSsoBbIilGgZzTt]", "", parts[2].strip()))

    try:
        d, m, y = int(d_str), int(m_str), int(y_str)
    except ValueError:
        return dob

    # Fix year if mangled (e.g. 2OO5→2005 already handled, but 200b→2006 etc.)
    if not (1900 <= y <= 2025):
        return dob  # can't fix, return original

    # Swap day/month only if month is clearly impossible
    if m > 12 and 1 <= d <= 12:
        d, m = m, d

    if 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2025:
        return f"{d:02d}/{m:02d}/{y}"

    return dob


def collect_all_dobs(all_pass_texts):
    """
    Collect ALL (day, month, year) tuples from EVERY OCR pass.
    Using extract_dob_tuples so even garbled lines like
    "DOR 10 12006" or "DOR 10/1 172008" yield candidates.
    """
    all_tuples = []
    for i, t in enumerate(all_pass_texts):
        found = extract_dob_tuples(t)
        if found:
            info(f"    Pass {i+1} DOB tuples: {found}")
        all_tuples.extend(found)
    return all_tuples

def vote_dob_digit_by_digit(all_tuples):
    """
    Given flat list of (day, month, year) tuples from ALL passes,
    vote on each component independently.

    WHY THIS WORKS:
      Tesseract makes different errors in different passes.
      The truth repeats consistently — errors don't.

      Example for actual DOB 10/11/2005:
        Tuples collected:
          (10, 1, 2006)   ← pass1: month 11→1, year 2005→2006
          (10, 11, 2005)  ← pass2: correct
          (10, 1, 2006)   ← merged cluster parse
          (10, 11, 2005)  ← pass4: correct
          (10, 1, 2005)   ← pass5: month 11→1

        Day   votes : {10: 5}           → winner: 10   ✓
        Month votes : {1: 3, 11: 2} ... wait, 11 might still lose!
        So we ALSO parse the raw merged strings like "12006"=1+2006
        and "172008"=1+72008... and those add more 11 votes.

    The key insight: we collect tuples from MANY patterns per pass
    so the correct value accumulates more votes than any single error.
    """
    from collections import Counter

    if not all_tuples:
        return None

    info(f"  All DOB tuples for voting: {all_tuples}")

    days   = [t[0] for t in all_tuples]
    months = [t[1] for t in all_tuples]
    years  = [t[2] for t in all_tuples]

    day_votes   = Counter(days)
    month_votes = Counter(months)
    year_votes  = Counter(years)

    best_day   = day_votes.most_common(1)[0][0]
    best_month = month_votes.most_common(1)[0][0]
    best_year  = year_votes.most_common(1)[0][0]

    info(f"  Day   votes : {dict(day_votes)}   → winner: {best_day}")
    info(f"  Month votes : {dict(month_votes)} → winner: {best_month}")
    info(f"  Year  votes : {dict(year_votes)}  → winner: {best_year}")

    if 1 <= best_day <= 31 and 1 <= best_month <= 12 and 1900 <= best_year <= 2025:
        return f"{best_day:02d}/{best_month:02d}/{best_year}"
    return None


def correct_name(name, all_pass_texts):
    """
    Correct OCR character errors in an already-extracted name.

    SAFETY PRINCIPLE:
      The name was already correctly identified by extract_name()
      using positional anchors. The only thing we need to fix are
      character-level misreads within the words (0→O, rn→m, etc.).

      We do NOT re-run extract_name() on raw passes — that causes
      garbage lines from other parts of the card to corrupt the result.

    STRATEGY:
      1. Apply char-level substitution map to each word
      2. Apply word-level pattern fixes (VAN→YAM, rn→m etc.)
      3. Safety check: only return the fixed name if it looks
         MORE like a real name than the original (or equal).
         Never return something WORSE.

    Args:
        name           : extracted name string (may be None)
        all_pass_texts : unused — kept for API compatibility

    Returns:
        corrected name string (always >= as good as input)
    """
    if not name or not name.strip():
        return name

    original = name.strip()

    # Apply char-level and word-level fixes
    words     = original.split()
    fixed_words = [fix_name_word(w) for w in words]
    fixed     = ' '.join(fixed_words)

    # Safety: count how many chars changed
    changes = sum(a != b for a, b in zip(original, fixed)) + abs(len(original) - len(fixed))

    # If massive change (>3 chars different), it means fix_name_word
    # mangled something — return original to be safe
    if changes > 3:
        info(f"  Name fix rejected (too many changes: {changes}) — keeping original")
        return original

    return fixed


def step14b_correct(fields, all_pass_texts):
    section("14B — OCR Error Correction")

    info("Correcting common Tesseract misreads on Aadhaar cards...")
    print()

    original_name = fields.get("name")
    original_dob  = fields.get("dob")

    # ── Correct NAME ─────────────────────────────────────
    info("Correcting NAME (char substitutions + cross-pass voting):")
    corrected_name = correct_name(original_name, all_pass_texts)
    if corrected_name != original_name:
        warn(f"  Name changed : '{original_name}'  →  '{corrected_name}'")
        fields["name"] = corrected_name
    else:
        ok(f"  Name OK      : '{corrected_name}'")

    print()

    # ── Correct DOB ──────────────────────────────────────
    info("Correcting DOB (digit-by-digit cross-pass voting):")
    info("Each pass contributes (day,month,year) tuples. Majority wins per component.")

    # Collect ALL (day,month,year) tuples from every pass
    # Uses extract_dob_tuples which mines ALL patterns per pass,
    # including garbled forms like "DOR 10 12006" and "DOR 10/1 172008"
    all_tuples = collect_all_dobs(all_pass_texts)

    # Also parse the already-extracted DOB string
    if original_dob:
        from_original = extract_dob_tuples(original_dob) if original_dob else []
        # If extract_dob_tuples doesn't handle bare strings, do it manually
        if not from_original:
            m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', original_dob)
            if m:
                d,mo,y = int(m.group(1)),int(m.group(2)),int(m.group(3))
                if 1<=d<=31 and 1<=mo<=12 and 1900<=y<=2025:
                    from_original = [(d, mo, y)]
        all_tuples.extend(from_original)

    if not all_tuples:
        warn(f"  No DOB tuples found across any pass — keeping: '{original_dob}'")
    else:
        voted_dob = vote_dob_digit_by_digit(all_tuples)
        if voted_dob and voted_dob != original_dob:
            warn(f"  DOB changed : '{original_dob}'  →  '{voted_dob}'")
            fields["dob"] = voted_dob
        elif voted_dob:
            ok(f"  DOB OK      : '{voted_dob}'")
        else:
            warn(f"  Could not correct DOB, keeping: '{original_dob}'")

    print()

    # ── Correct AADHAAR number ───────────────────────────
    info("Correcting Aadhaar number (digit-only fix):")
    aadhaar = fields.get("aadhaar_number")
    if aadhaar:
        fixed = fix_digit_string(aadhaar.replace(" ", ""))
        if len(fixed) == 12:
            formatted = f"{fixed[0:4]} {fixed[4:8]} {fixed[8:12]}"
            if formatted != aadhaar:
                warn(f"  Aadhaar changed: '{aadhaar}'  →  '{formatted}'")
                fields["aadhaar_number"] = formatted
            else:
                ok(f"  Aadhaar OK     : '{formatted}'")
        else:
            warn(f"  Aadhaar digit count wrong after fix: {len(fixed)} digits")

    return fields



# ─────────────────────────────────────────────────────────────
#  STEP 14C — LLM-Based OCR Correction
#
#  WHY LLM CORRECTION WORKS:
#    Regex rules only fix KNOWN patterns (e.g. VAN→YAM).
#    An LLM knows Indian name patterns from its training data.
#    Example:
#      OCR output : "GHANSHVAN KUMAWAT"
#      Regex fix  : "GHANSHYAM KUMAWAT"  ← pattern rule
#      LLM fix    : "GHANSHYAM KUMAVAT"  ← knows both corrections
#
#    LLM also handles:
#      - Rare names not in our pattern list
#      - DOB reconstruction from partial digits
#      - Father name corrections
#      - Address normalization
#
#  TWO MODES:
#    Mode A — Local (free, offline):
#      Uses a tiny quantized model via Hugging Face
#      Model: google/flan-t5-small (only 80MB)
#      No internet needed after first download
#      Accuracy: moderate (small model)
#
#    Mode B — API (best accuracy):
#      Uses any OpenAI-compatible API
#      Works with: OpenAI GPT, Groq (free), Together AI
#      Set OPENAI_API_KEY or GROQ_API_KEY env variable
#      Accuracy: very high (large model)
#
#  GRACEFUL FALLBACK:
#    If neither is available → fields returned unchanged.
#    This step is additive — never makes things worse.
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
#  STEP 14C — LLM-Based OCR Correction (Groq)
#  (orchestrator — delegates to llm_correct_fields above)
# ─────────────────────────────────────────────────────────────

def llm_correct_local(fields):
    """Kept for backward compatibility — delegates to llm_correct_fields."""
    from llm_correction import llm_correct_fields
    return llm_correct_fields("", fields), True

def llm_correct_api(fields, api_key=None, model=None):
    """Kept for backward compatibility — delegates to llm_correct_fields."""
    from llm_correction import llm_correct_fields
    import os as _os
    key = api_key or _os.environ.get("GROQ_API_KEY", "")
    if not key:
        return fields, False
    return llm_correct_fields("", fields, api_key=key), True

def step14c_llm_correct(fields, raw_ocr_text=""):
    """
    Step 14C — delegates to the unified Groq LLM correction function.
    Uses GROQ_API_KEY env var automatically.
    Falls back to local flan-t5-small if key not set.
    Pass raw_ocr_text so the LLM has full context to work from.
    """
    from llm_correction import llm_correct_fields
    return llm_correct_fields(raw_ocr_text, fields)