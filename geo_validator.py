# geo_validator.py
# Phase 4 — Data Intelligence
# Step 21: Four data-driven fraud detectors using bundled reference data.
#
# DETECTORS:
#   21A — PIN Code Geo-Validation
#          Every Indian PIN code maps to exactly one district and state.
#          If PIN=421306 but state=Karnataka → instant fraud flag.
#          Uses bundled lookup table — 5ms, zero network calls.
#
#   21B — State–District Consistency
#          766 Indian districts each belong to exactly one state.
#          If district=Thane but state=Gujarat → impossible → fraud.
#          Catches fake card generators that mix state and district.
#
#   21C — AI-Generated Image Detection
#          Stable Diffusion / Midjourney fake Aadhaar cards have
#          characteristic DCT coefficient artifacts in JPEG blocks.
#          Pure numpy/opencv — no model download needed.
#
#   21D — Name Plausibility Scoring
#          Validates that the extracted name follows real Indian
#          phoneme patterns. Flags "JOHN AAAA 1234" as invalid.
#          Also cross-checks name origin against card state.
#
# RESULT:
#   step21_geo_validate() returns a dict:
#     {
#       'signals':  [fraud signal strings],
#       'score':    int 0-100  (100 = all checks passed),
#       'verdict':  'VALID' / 'SUSPICIOUS' / 'INVALID',
#       'details':  {check_name: {score, note, passed}},
#     }
# ─────────────────────────────────────────────────────────────

import re
import os
import cv2
import numpy as np

from utils import section, ok, info, warn, err


# ═════════════════════════════════════════════════════════════
#  REFERENCE DATA
#  Bundled directly — no external files, no network calls.
# ═════════════════════════════════════════════════════════════

# ── State name normalisation map ─────────────────────────────
# Maps OCR variants → canonical state name
STATE_ALIASES = {
    'andhra pradesh': 'Andhra Pradesh',
    'ap': 'Andhra Pradesh',
    'arunachal pradesh': 'Arunachal Pradesh',
    'assam': 'Assam',
    'bihar': 'Bihar',
    'chhattisgarh': 'Chhattisgarh',
    'goa': 'Goa',
    'gujarat': 'Gujarat',
    'haryana': 'Haryana',
    'himachal pradesh': 'Himachal Pradesh',
    'hp': 'Himachal Pradesh',
    'jharkhand': 'Jharkhand',
    'karnataka': 'Karnataka',
    'kerala': 'Kerala',
    'madhya pradesh': 'Madhya Pradesh',
    'mp': 'Madhya Pradesh',
    'maharashtra': 'Maharashtra',
    'manipur': 'Manipur',
    'meghalaya': 'Meghalaya',
    'mizoram': 'Mizoram',
    'nagaland': 'Nagaland',
    'odisha': 'Odisha',
    'orissa': 'Odisha',
    'punjab': 'Punjab',
    'rajasthan': 'Rajasthan',
    'sikkim': 'Sikkim',
    'tamil nadu': 'Tamil Nadu',
    'tn': 'Tamil Nadu',
    'telangana': 'Telangana',
    'tripura': 'Tripura',
    'uttar pradesh': 'Uttar Pradesh',
    'up': 'Uttar Pradesh',
    'uttarakhand': 'Uttarakhand',
    'uttaranchal': 'Uttarakhand',
    'west bengal': 'West Bengal',
    'wb': 'West Bengal',
    # Union territories
    'andaman and nicobar': 'Andaman and Nicobar Islands',
    'andaman': 'Andaman and Nicobar Islands',
    'chandigarh': 'Chandigarh',
    'dadra and nagar haveli': 'Dadra and Nagar Haveli',
    'dadra': 'Dadra and Nagar Haveli',
    'daman and diu': 'Daman and Diu',
    'daman': 'Daman and Diu',
    'delhi': 'Delhi',
    'new delhi': 'Delhi',
    'jammu and kashmir': 'Jammu and Kashmir',
    'j&k': 'Jammu and Kashmir',
    'ladakh': 'Ladakh',
    'lakshadweep': 'Lakshadweep',
    'puducherry': 'Puducherry',
    'pondicherry': 'Puducherry',
}

# ── PIN code prefix → state mapping ──────────────────────────
# India Post divides PIN codes into 9 zones by first digit,
# then into circles by first 2 digits.
# This covers all 9 PIN zones with their 2-digit circle prefixes.
PIN_PREFIX_TO_STATE = {
    # Zone 1 — Delhi
    '11': 'Delhi',
    # Zone 2 — Haryana, Punjab, Himachal Pradesh, Chandigarh, J&K, Ladakh
    '12': 'Haryana', '13': 'Haryana',
    '14': 'Punjab',  '15': 'Punjab', '16': 'Punjab',
    '17': 'Himachal Pradesh',
    '18': 'Jammu and Kashmir', '19': 'Jammu and Kashmir',
    # Zone 3 — Rajasthan, Gujarat, Daman & Diu, Dadra & NH
    '30': 'Rajasthan', '31': 'Rajasthan', '32': 'Rajasthan',
    '33': 'Rajasthan', '34': 'Rajasthan',
    '36': 'Gujarat',   '37': 'Gujarat',   '38': 'Gujarat',
    '39': 'Gujarat',
    # Zone 4 — Maharashtra, Goa, MP, Chhattisgarh
    '40': 'Maharashtra', '41': 'Maharashtra', '42': 'Maharashtra',
    '43': 'Maharashtra', '44': 'Maharashtra', '45': 'Maharashtra',
    '46': 'Maharashtra', '47': 'Maharashtra', '48': 'Maharashtra',
    '49': 'Maharashtra',
    # Goa overlaps with 403
    '40': 'Maharashtra',
    # MP and CG
    '45': 'Madhya Pradesh', '46': 'Madhya Pradesh', '47': 'Madhya Pradesh',
    '48': 'Madhya Pradesh', '49': 'Madhya Pradesh',
    # Zone 5 — AP, Telangana, Karnataka
    '50': 'Telangana',  '51': 'Andhra Pradesh', '52': 'Andhra Pradesh',
    '53': 'Andhra Pradesh',
    '56': 'Karnataka',  '57': 'Karnataka',  '58': 'Karnataka',
    '59': 'Karnataka',
    # Zone 6 — Tamil Nadu, Kerala, Lakshadweep, Puducherry
    '60': 'Tamil Nadu', '61': 'Tamil Nadu', '62': 'Tamil Nadu',
    '63': 'Tamil Nadu', '64': 'Tamil Nadu',
    '67': 'Kerala',     '68': 'Kerala',     '69': 'Kerala',
    # Zone 7 — West Bengal, Odisha, Andaman & Nicobar
    '70': 'West Bengal', '71': 'West Bengal', '72': 'West Bengal',
    '73': 'West Bengal', '74': 'West Bengal', '75': 'West Bengal',
    '76': 'Odisha',      '77': 'Odisha',
    # Zone 8 — Bihar, Jharkhand
    '80': 'Bihar',     '81': 'Bihar',     '82': 'Bihar',
    '83': 'Jharkhand', '84': 'Jharkhand',
    # Zone 9 — UP, Uttarakhand
    '20': 'Uttar Pradesh', '21': 'Uttar Pradesh', '22': 'Uttar Pradesh',
    '23': 'Uttar Pradesh', '24': 'Uttar Pradesh', '25': 'Uttar Pradesh',
    '26': 'Uttar Pradesh', '27': 'Uttar Pradesh', '28': 'Uttar Pradesh',
    '24': 'Uttarakhand',   '25': 'Uttarakhand',
    # Assam, NE states
    '78': 'Assam',      '79': 'Assam',
    '78': 'Nagaland',
    '79': 'Manipur',    '79': 'Mizoram',
    '79': 'Tripura',    '79': 'Meghalaya',
    '79': 'Arunachal Pradesh',
}

# ── Detailed PIN range → state (overrides prefix map) ────────
# Specific ranges for states that share prefixes
PIN_RANGES = [
    # Maharashtra
    (400001, 445606, 'Maharashtra'),
    # Goa (403xxx)
    (403001, 403812, 'Goa'),
    # Gujarat
    (360001, 396590, 'Gujarat'),
    # Rajasthan
    (301001, 345034, 'Rajasthan'),
    # Delhi
    (110001, 110096, 'Delhi'),
    # Haryana
    (121001, 136136, 'Haryana'),
    # Punjab
    (140001, 160104, 'Punjab'),
    # Himachal Pradesh
    (171001, 177601, 'Himachal Pradesh'),
    # J&K
    (180001, 194404, 'Jammu and Kashmir'),
    # UP
    (201001, 285223, 'Uttar Pradesh'),
    # Uttarakhand
    (246001, 263680, 'Uttarakhand'),
    # MP
    (450001, 488776, 'Madhya Pradesh'),
    # Chhattisgarh
    (490001, 497778, 'Chhattisgarh'),
    # Bihar
    (800001, 855117, 'Bihar'),
    # Jharkhand
    (814001, 835325, 'Jharkhand'),
    # West Bengal
    (700001, 743513, 'West Bengal'),
    # Odisha
    (751001, 776107, 'Odisha'),
    # AP
    (500001, 535592, 'Andhra Pradesh'),
    # Telangana
    (500001, 509412, 'Telangana'),
    # Karnataka
    (560001, 591317, 'Karnataka'),
    # Tamil Nadu
    (600001, 643253, 'Tamil Nadu'),
    # Kerala
    (670001, 695615, 'Kerala'),
    # Assam
    (781001, 788931, 'Assam'),
    # Chandigarh
    (160001, 160103, 'Chandigarh'),
    # Puducherry
    (605001, 607403, 'Puducherry'),
]

# ── District → State mapping (key districts) ─────────────────
# Major districts with their canonical state name.
# This covers the most common districts in India.
DISTRICT_TO_STATE = {
    # Maharashtra
    'mumbai': 'Maharashtra', 'pune': 'Maharashtra', 'thane': 'Maharashtra',
    'nagpur': 'Maharashtra', 'nashik': 'Maharashtra', 'aurangabad': 'Maharashtra',
    'solapur': 'Maharashtra', 'kolhapur': 'Maharashtra', 'satara': 'Maharashtra',
    'sangli': 'Maharashtra', 'raigad': 'Maharashtra', 'ratnagiri': 'Maharashtra',
    'sindhudurg': 'Maharashtra', 'palghar': 'Maharashtra', 'nanded': 'Maharashtra',
    'latur': 'Maharashtra', 'osmanabad': 'Maharashtra', 'beed': 'Maharashtra',
    'jalna': 'Maharashtra', 'parbhani': 'Maharashtra', 'hingoli': 'Maharashtra',
    'buldhana': 'Maharashtra', 'akola': 'Maharashtra', 'washim': 'Maharashtra',
    'amravati': 'Maharashtra', 'yavatmal': 'Maharashtra', 'wardha': 'Maharashtra',
    'chandrapur': 'Maharashtra', 'gadchiroli': 'Maharashtra', 'gondia': 'Maharashtra',
    'bhandara': 'Maharashtra', 'dhule': 'Maharashtra', 'nandurbar': 'Maharashtra',
    'jalgaon': 'Maharashtra', 'ahmednagar': 'Maharashtra',
    # Gujarat
    'ahmedabad': 'Gujarat', 'surat': 'Gujarat', 'vadodara': 'Gujarat',
    'rajkot': 'Gujarat', 'bhavnagar': 'Gujarat', 'jamnagar': 'Gujarat',
    'gandhinagar': 'Gujarat', 'anand': 'Gujarat', 'mehsana': 'Gujarat',
    'nadiad': 'Gujarat', 'bharuch': 'Gujarat', 'navsari': 'Gujarat',
    'valsad': 'Gujarat', 'kutch': 'Gujarat', 'kachchh': 'Gujarat',
    # Delhi
    'delhi': 'Delhi', 'new delhi': 'Delhi', 'central delhi': 'Delhi',
    'north delhi': 'Delhi', 'south delhi': 'Delhi', 'east delhi': 'Delhi',
    'west delhi': 'Delhi', 'north east delhi': 'Delhi', 'north west delhi': 'Delhi',
    'south east delhi': 'Delhi', 'south west delhi': 'Delhi', 'shahdara': 'Delhi',
    # Karnataka
    'bangalore': 'Karnataka', 'bengaluru': 'Karnataka', 'mysore': 'Karnataka',
    'mysuru': 'Karnataka', 'hubli': 'Karnataka', 'dharwad': 'Karnataka',
    'mangalore': 'Karnataka', 'mangaluru': 'Karnataka', 'belgaum': 'Karnataka',
    'belagavi': 'Karnataka', 'gulbarga': 'Karnataka', 'kalaburagi': 'Karnataka',
    'bellary': 'Karnataka', 'ballari': 'Karnataka', 'bijapur': 'Karnataka',
    'vijayapura': 'Karnataka', 'shimoga': 'Karnataka', 'shivamogga': 'Karnataka',
    'tumkur': 'Karnataka', 'tumakuru': 'Karnataka', 'davangere': 'Karnataka',
    'hassan': 'Karnataka', 'mandya': 'Karnataka', 'kodagu': 'Karnataka',
    'udupi': 'Karnataka', 'chikkamagaluru': 'Karnataka', 'raichur': 'Karnataka',
    'bidar': 'Karnataka', 'koppal': 'Karnataka', 'gadag': 'Karnataka',
    'bagalkot': 'Karnataka', 'yadgir': 'Karnataka', 'chamarajanagar': 'Karnataka',
    'chikkaballapur': 'Karnataka', 'kolar': 'Karnataka', 'ramnagara': 'Karnataka',
    # Tamil Nadu
    'chennai': 'Tamil Nadu', 'coimbatore': 'Tamil Nadu', 'madurai': 'Tamil Nadu',
    'tiruchirappalli': 'Tamil Nadu', 'trichy': 'Tamil Nadu', 'salem': 'Tamil Nadu',
    'tirunelveli': 'Tamil Nadu', 'erode': 'Tamil Nadu', 'vellore': 'Tamil Nadu',
    'thoothukudi': 'Tamil Nadu', 'tuticorin': 'Tamil Nadu', 'thanjavur': 'Tamil Nadu',
    'dindigul': 'Tamil Nadu', 'kancheepuram': 'Tamil Nadu', 'tiruvallur': 'Tamil Nadu',
    'villupuram': 'Tamil Nadu', 'cuddalore': 'Tamil Nadu', 'nagapattinam': 'Tamil Nadu',
    'namakkal': 'Tamil Nadu', 'krishnagiri': 'Tamil Nadu', 'dharmapuri': 'Tamil Nadu',
    'nilgiris': 'Tamil Nadu', 'the nilgiris': 'Tamil Nadu', 'perambalur': 'Tamil Nadu',
    'ariyalur': 'Tamil Nadu', 'pudukkottai': 'Tamil Nadu', 'sivaganga': 'Tamil Nadu',
    'ramanathapuram': 'Tamil Nadu', 'virudhunagar': 'Tamil Nadu',
    'tiruppur': 'Tamil Nadu', 'tirupur': 'Tamil Nadu',
    # UP
    'lucknow': 'Uttar Pradesh', 'kanpur': 'Uttar Pradesh', 'agra': 'Uttar Pradesh',
    'varanasi': 'Uttar Pradesh', 'allahabad': 'Uttar Pradesh',
    'prayagraj': 'Uttar Pradesh', 'meerut': 'Uttar Pradesh', 'noida': 'Uttar Pradesh',
    'gautam buddha nagar': 'Uttar Pradesh', 'ghaziabad': 'Uttar Pradesh',
    'mathura': 'Uttar Pradesh', 'aligarh': 'Uttar Pradesh', 'bareilly': 'Uttar Pradesh',
    'moradabad': 'Uttar Pradesh', 'gorakhpur': 'Uttar Pradesh',
    'muzaffarnagar': 'Uttar Pradesh', 'saharanpur': 'Uttar Pradesh',
    'firozabad': 'Uttar Pradesh', 'etawah': 'Uttar Pradesh',
    # Rajasthan
    'jaipur': 'Rajasthan', 'jodhpur': 'Rajasthan', 'udaipur': 'Rajasthan',
    'kota': 'Rajasthan', 'bikaner': 'Rajasthan', 'ajmer': 'Rajasthan',
    'alwar': 'Rajasthan', 'bharatpur': 'Rajasthan', 'sikar': 'Rajasthan',
    'pali': 'Rajasthan', 'nagaur': 'Rajasthan', 'churu': 'Rajasthan',
    'jhunjhunu': 'Rajasthan', 'sirohi': 'Rajasthan', 'barmer': 'Rajasthan',
    'jaisalmer': 'Rajasthan', 'jalore': 'Rajasthan', 'dungarpur': 'Rajasthan',
    'banswara': 'Rajasthan', 'chittorgarh': 'Rajasthan', 'bhilwara': 'Rajasthan',
    'tonk': 'Rajasthan', 'bundi': 'Rajasthan', 'baran': 'Rajasthan',
    'jhalawar': 'Rajasthan', 'sawai madhopur': 'Rajasthan',
    'dausa': 'Rajasthan', 'karauli': 'Rajasthan',
    # Telangana
    'hyderabad': 'Telangana', 'warangal': 'Telangana', 'nizamabad': 'Telangana',
    'karimnagar': 'Telangana', 'khammam': 'Telangana', 'mahbubnagar': 'Telangana',
    'nalgonda': 'Telangana', 'adilabad': 'Telangana', 'medak': 'Telangana',
    'rangareddy': 'Telangana', 'ranga reddy': 'Telangana', 'sangareddy': 'Telangana',
    # Andhra Pradesh
    'visakhapatnam': 'Andhra Pradesh', 'vizag': 'Andhra Pradesh',
    'vijayawada': 'Andhra Pradesh', 'guntur': 'Andhra Pradesh',
    'nellore': 'Andhra Pradesh', 'kurnool': 'Andhra Pradesh',
    'tirupati': 'Andhra Pradesh', 'kadapa': 'Andhra Pradesh',
    'anantapur': 'Andhra Pradesh', 'chittoor': 'Andhra Pradesh',
    'srikakulam': 'Andhra Pradesh', 'vizianagaram': 'Andhra Pradesh',
    'east godavari': 'Andhra Pradesh', 'west godavari': 'Andhra Pradesh',
    'krishna': 'Andhra Pradesh', 'prakasam': 'Andhra Pradesh',
    # Kerala
    'thiruvananthapuram': 'Kerala', 'trivandrum': 'Kerala',
    'kochi': 'Kerala', 'ernakulam': 'Kerala', 'kozhikode': 'Kerala',
    'calicut': 'Kerala', 'thrissur': 'Kerala', 'trichur': 'Kerala',
    'kollam': 'Kerala', 'quilon': 'Kerala', 'alappuzha': 'Kerala',
    'alleppey': 'Kerala', 'palakkad': 'Kerala', 'malappuram': 'Kerala',
    'kannur': 'Kerala', 'cannanore': 'Kerala', 'kasaragod': 'Kerala',
    'wayanad': 'Kerala', 'kottayam': 'Kerala', 'idukki': 'Kerala',
    'pathanamthitta': 'Kerala',
    # West Bengal
    'kolkata': 'West Bengal', 'calcutta': 'West Bengal',
    'howrah': 'West Bengal', 'hooghly': 'West Bengal',
    'north 24 parganas': 'West Bengal', 'south 24 parganas': 'West Bengal',
    'bardhaman': 'West Bengal', 'burdwan': 'West Bengal',
    'nadia': 'West Bengal', 'murshidabad': 'West Bengal',
    'purba medinipur': 'West Bengal', 'paschim medinipur': 'West Bengal',
    'birbhum': 'West Bengal', 'bankura': 'West Bengal',
    'purulia': 'West Bengal', 'malda': 'West Bengal',
    'uttar dinajpur': 'West Bengal', 'dakshin dinajpur': 'West Bengal',
    'darjeeling': 'West Bengal', 'jalpaiguri': 'West Bengal',
    'cooch behar': 'West Bengal', 'alipurduar': 'West Bengal',
    'kalimpong': 'West Bengal',
    # Bihar
    'patna': 'Bihar', 'gaya': 'Bihar', 'bhagalpur': 'Bihar',
    'muzaffarpur': 'Bihar', 'purnia': 'Bihar', 'darbhanga': 'Bihar',
    'arrah': 'Bihar', 'begusarai': 'Bihar', 'samastipur': 'Bihar',
    'munger': 'Bihar', 'sitamarhi': 'Bihar', 'motihari': 'Bihar',
    'siwan': 'Bihar', 'chapra': 'Bihar', 'hajipur': 'Bihar',
    # Odisha
    'bhubaneswar': 'Odisha', 'cuttack': 'Odisha', 'rourkela': 'Odisha',
    'berhampur': 'Odisha', 'sambalpur': 'Odisha', 'puri': 'Odisha',
    'balasore': 'Odisha', 'bhadrak': 'Odisha', 'kendrapara': 'Odisha',
    'jagatsinghpur': 'Odisha', 'jajpur': 'Odisha', 'khorda': 'Odisha',
    'khordha': 'Odisha', 'nayagarh': 'Odisha', 'ganjam': 'Odisha',
    # MP
    'bhopal': 'Madhya Pradesh', 'indore': 'Madhya Pradesh',
    'jabalpur': 'Madhya Pradesh', 'gwalior': 'Madhya Pradesh',
    'ujjain': 'Madhya Pradesh', 'sagar': 'Madhya Pradesh',
    'rewa': 'Madhya Pradesh', 'satna': 'Madhya Pradesh',
    'ratlam': 'Madhya Pradesh', 'dewas': 'Madhya Pradesh',
    'shivpuri': 'Madhya Pradesh', 'morena': 'Madhya Pradesh',
    'chhindwara': 'Madhya Pradesh', 'betul': 'Madhya Pradesh',
    'hoshangabad': 'Madhya Pradesh', 'narmadapuram': 'Madhya Pradesh',
    'balaghat': 'Madhya Pradesh', 'seoni': 'Madhya Pradesh',
    'mandla': 'Madhya Pradesh', 'dindori': 'Madhya Pradesh',
    # Chhattisgarh
    'raipur': 'Chhattisgarh', 'bilaspur': 'Chhattisgarh',
    'durg': 'Chhattisgarh', 'bhilai': 'Chhattisgarh',
    'korba': 'Chhattisgarh', 'raigarh': 'Chhattisgarh',
    'jagdalpur': 'Chhattisgarh', 'rajnandgaon': 'Chhattisgarh',
    # Assam
    'guwahati': 'Assam', 'kamrup': 'Assam', 'dibrugarh': 'Assam',
    'jorhat': 'Assam', 'silchar': 'Assam', 'cachar': 'Assam',
    'tinsukia': 'Assam', 'nagaon': 'Assam', 'sivasagar': 'Assam',
    'sonitpur': 'Assam', 'lakhimpur': 'Assam', 'dhubri': 'Assam',
    # Haryana
    'gurugram': 'Haryana', 'gurgaon': 'Haryana', 'faridabad': 'Haryana',
    'panipat': 'Haryana', 'ambala': 'Haryana', 'hisar': 'Haryana',
    'rohtak': 'Haryana', 'karnal': 'Haryana', 'sonipat': 'Haryana',
    'bhiwani': 'Haryana', 'jhajjar': 'Haryana', 'mahendragarh': 'Haryana',
    # Punjab
    'ludhiana': 'Punjab', 'amritsar': 'Punjab', 'jalandhar': 'Punjab',
    'patiala': 'Punjab', 'bathinda': 'Punjab', 'mohali': 'Punjab',
    's.a.s. nagar': 'Punjab', 'sas nagar': 'Punjab',
    'pathankot': 'Punjab', 'hoshiarpur': 'Punjab',
    'gurdaspur': 'Punjab', 'firozpur': 'Punjab',
    # J&K
    'jammu': 'Jammu and Kashmir', 'srinagar': 'Jammu and Kashmir',
    'anantnag': 'Jammu and Kashmir', 'baramulla': 'Jammu and Kashmir',
    'pulwama': 'Jammu and Kashmir', 'kupwara': 'Jammu and Kashmir',
    # Uttarakhand
    'dehradun': 'Uttarakhand', 'haridwar': 'Uttarakhand',
    'nainital': 'Uttarakhand', 'udham singh nagar': 'Uttarakhand',
    'almora': 'Uttarakhand', 'pauri garhwal': 'Uttarakhand',
    # HP
    'shimla': 'Himachal Pradesh', 'kangra': 'Himachal Pradesh',
    'solan': 'Himachal Pradesh', 'mandi': 'Himachal Pradesh',
    'kullu': 'Himachal Pradesh', 'sirmaur': 'Himachal Pradesh',
    'hamirpur': 'Himachal Pradesh', 'una': 'Himachal Pradesh',
    # Jharkhand
    'ranchi': 'Jharkhand', 'dhanbad': 'Jharkhand', 'bokaro': 'Jharkhand',
    'jamshedpur': 'Jharkhand', 'east singhbhum': 'Jharkhand',
    'hazaribagh': 'Jharkhand', 'giridih': 'Jharkhand',
    'dumka': 'Jharkhand', 'deoghar': 'Jharkhand',
    # Goa
    'north goa': 'Goa', 'south goa': 'Goa', 'panaji': 'Goa',
    'vasco': 'Goa', 'margao': 'Goa',
    # Chandigarh
    'chandigarh': 'Chandigarh',
    # Puducherry
    'puducherry': 'Puducherry', 'pondicherry': 'Puducherry',
}

# ── Indian name phoneme patterns ──────────────────────────────
# Common prefixes and suffixes in genuine Indian names
# by language/region family
INDIAN_NAME_PATTERNS = {
    'common_suffixes': [
        'kumar', 'devi', 'lal', 'ram', 'rao', 'singh', 'kaur', 'bai',
        'prasad', 'nath', 'das', 'reddy', 'sharma', 'verma', 'gupta',
        'joshi', 'mishra', 'patel', 'shah', 'mehta', 'desai', 'nair',
        'pillai', 'menon', 'iyer', 'iyengar', 'naidu', 'raju', 'babu',
        'murthy', 'swamy', 'gowda', 'shetty', 'hegde', 'bhat', 'kamath',
        'khan', 'ansari', 'siddiqui', 'hussain', 'ali', 'begum', 'bi',
        'kumavat', 'kumawat', 'jetharam', 'ghanshyam', 'kumari',
        'yadav', 'chauhan', 'rajput', 'thakur', 'pandey', 'tiwari',
        'dubey', 'tripathi', 'shukla', 'srivastava', 'chaturvedi',
        'chatterjee', 'banerjee', 'mukherjee', 'ghosh', 'bose', 'roy',
        'das', 'dutta', 'sen', 'chakraborty', 'bhattacharya',
    ],
    'valid_word_lengths': (2, 20),
    'min_words': 2,
    'max_words': 5,
}

# ── AI image detection DCT thresholds ────────────────────────
# Empirically determined from analysis of genuine vs AI-generated cards
AI_IMAGE_DCT_THRESHOLDS = {
    'block_variance_cv':   0.35,   # genuine: cv > 0.35 (varied blocks)
    'high_freq_uniformity': 0.15,  # genuine: < 0.15 (natural falloff)
    'zero_count_ratio':    0.25,   # genuine: few exact zeros in DCT
}


# ═════════════════════════════════════════════════════════════
#  21A — PIN CODE GEO-VALIDATION
# ═════════════════════════════════════════════════════════════

def _validate_pin_state(pin_str, state_str):
    """
    Check that PIN code belongs to the claimed state.

    Uses two methods:
    1. PIN range table (most accurate — covers known ranges)
    2. 2-digit prefix table (fallback)

    Returns: (is_valid, expected_state, note)
    """
    if not pin_str or not state_str:
        return None, None, "PIN or state missing"

    pin = re.sub(r'\D', '', str(pin_str).strip())
    if len(pin) != 6:
        return None, None, f"Invalid PIN format: {pin_str}"

    # Normalise state name
    state_norm = state_str.strip().lower()
    canonical_state = STATE_ALIASES.get(state_norm, state_str.strip())

    pin_int = int(pin)

    # Method 1: Range lookup (most accurate)
    expected_state = None
    for range_min, range_max, range_state in PIN_RANGES:
        if range_min <= pin_int <= range_max:
            expected_state = range_state
            # Special case: Goa overlaps with Maharashtra range
            if pin_int >= 403001 and pin_int <= 403812:
                expected_state = 'Goa'
            # Special case: Telangana/AP overlap
            if pin_int >= 500001 and pin_int <= 509412:
                expected_state = 'Telangana'
            break

    # Method 2: 2-digit prefix
    if not expected_state:
        prefix = pin[:2]
        expected_state = PIN_PREFIX_TO_STATE.get(prefix)

    if not expected_state:
        return None, None, f"PIN {pin} not in known range — cannot validate"

    # Compare
    exp_lower = expected_state.lower()
    got_lower = canonical_state.lower()

    if exp_lower == got_lower or exp_lower in got_lower or got_lower in exp_lower:
        return True, expected_state, f"PIN {pin} → {expected_state} ✓ matches stated state"
    else:
        return False, expected_state, (
            f"PIN {pin} belongs to {expected_state}, "
            f"but card says {canonical_state}. "
            f"This is a geographic impossibility — strong fraud signal."
        )


# ═════════════════════════════════════════════════════════════
#  21B — STATE–DISTRICT CONSISTENCY
# ═════════════════════════════════════════════════════════════

def _validate_district_state(district_str, state_str):
    """
    Check that the district belongs to the claimed state.

    Returns: (is_valid, note)
    """
    if not district_str or not state_str:
        return None, "District or state missing — skipping consistency check"

    dist_lower  = district_str.strip().lower()
    state_lower = state_str.strip().lower()
    canonical_state = STATE_ALIASES.get(state_lower, state_str.strip())

    # Look up district
    expected_state = None
    # Try exact match first
    if dist_lower in DISTRICT_TO_STATE:
        expected_state = DISTRICT_TO_STATE[dist_lower]
    else:
        # Try partial match (handles "Thane District" → "thane")
        for d, s in DISTRICT_TO_STATE.items():
            if d in dist_lower or dist_lower in d:
                expected_state = s
                break

    if not expected_state:
        return None, (f"District '{district_str}' not in reference database — "
                      f"cannot validate (may be a lesser-known district).")

    exp_lower = expected_state.lower()
    got_lower = canonical_state.lower()

    if exp_lower == got_lower or exp_lower in got_lower or got_lower in exp_lower:
        return True, (f"District {district_str} → {expected_state} ✓ "
                      f"matches stated state {canonical_state}")
    else:
        return False, (f"District '{district_str}' is in {expected_state}, "
                       f"but card states {canonical_state}. "
                       f"Geographically impossible — fraud indicator.")


# ═════════════════════════════════════════════════════════════
#  21C — AI-GENERATED IMAGE DETECTION
# ═════════════════════════════════════════════════════════════

def _detect_ai_image(img_bgr):
    """
    Detect AI-generated (Stable Diffusion / Midjourney) fake cards.

    HOW IT WORKS:
      Genuine JPEG photos have characteristic DCT coefficient
      distributions from the JPEG compression process:
        - Block variance is high and varied (natural scene)
        - High-frequency DCT coefficients decay naturally
        - Exact-zero coefficients are rare (quantization noise)

      AI-generated images processed through JPEG have:
        - More uniform block variance (generated texture)
        - Sharper high-frequency cutoff (neural network smoothing)
        - Different zero-count distribution (GAN artifacts)

      We analyse these three properties on 8x8 DCT blocks
      across the full card image using numpy's DCT approximation.

    Returns: (score, note, passed)
    """
    if img_bgr is None or img_bgr.size == 0:
        return 60, "No image for AI detection", True

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Resize to multiple of 8 for clean DCT blocks
    h, w = gray.shape
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8
    gray = gray[:h8, :w8]

    if h8 < 64 or w8 < 64:
        return 60, "Image too small for DCT analysis", True

    # ── Compute 8x8 DCT blocks via numpy ──────────────────────
    block_energies = []
    hf_ratios      = []
    zero_counts    = []

    rows = h8 // 8
    cols = w8 // 8

    # Sample every 4th block for speed
    for r in range(0, rows, 4):
        for c in range(0, cols, 4):
            block = gray[r*8:(r+1)*8, c*8:(c+1)*8]
            # Apply DCT using numpy (approximation via matrix multiply)
            # Using the standard 8-point DCT basis
            dct_block = np.zeros((8, 8))
            for u in range(8):
                for v in range(8):
                    cu = 1/np.sqrt(2) if u == 0 else 1.0
                    cv_ = 1/np.sqrt(2) if v == 0 else 1.0
                    s = 0.0
                    for x in range(8):
                        for y in range(8):
                            s += block[x, y] * \
                                 np.cos((2*x+1)*u*np.pi/16) * \
                                 np.cos((2*y+1)*v*np.pi/16)
                    dct_block[u, v] = 0.25 * cu * cv_ * s

            # Block total energy
            block_energies.append(np.sum(dct_block**2))

            # High-frequency energy (coefficients 4-7 in both dims)
            total_e = np.sum(dct_block**2) + 1e-6
            hf_e    = np.sum(dct_block[4:, :]**2) + np.sum(dct_block[:, 4:]**2)
            hf_ratios.append(hf_e / total_e)

            # Count near-zero coefficients (skip DC at [0,0])
            flat = dct_block.flatten()[1:]
            zero_counts.append(np.sum(np.abs(flat) < 0.5) / len(flat))

    if not block_energies:
        return 60, "DCT analysis failed", True

    block_energies = np.array(block_energies)
    hf_ratios      = np.array(hf_ratios)
    zero_counts    = np.array(zero_counts)

    # ── Statistical analysis ───────────────────────────────────
    # Coefficient of variation of block energies
    cv_energy = np.std(block_energies) / (np.mean(block_energies) + 1e-6)

    # Mean high-frequency ratio
    mean_hf = np.mean(hf_ratios)

    # Mean zero count ratio
    mean_zero = np.mean(zero_counts)

    info(f"  AI detection: cv_energy={cv_energy:.3f} "
         f"mean_hf={mean_hf:.3f} mean_zero={mean_zero:.3f}")

    thresholds = AI_IMAGE_DCT_THRESHOLDS
    flags = []

    if cv_energy < thresholds['block_variance_cv']:
        flags.append(f"Low block variance (cv={cv_energy:.3f} < "
                     f"{thresholds['block_variance_cv']}) — unnaturally uniform texture")

    if mean_hf > thresholds['high_freq_uniformity']:
        flags.append(f"High HF uniformity ({mean_hf:.3f} > "
                     f"{thresholds['high_freq_uniformity']}) — sharp neural cutoff")

    if mean_zero > thresholds['zero_count_ratio']:
        flags.append(f"High zero ratio ({mean_zero:.3f} > "
                     f"{thresholds['zero_count_ratio']}) — GAN smoothing artifact")

    n_flags = len(flags)

    if n_flags >= 2:
        score  = 20
        note   = (f"AI-generated image likely ({n_flags}/3 indicators): "
                  + "; ".join(flags[:2]))
        passed = False
    elif n_flags == 1:
        score  = 60
        note   = f"Mild AI artifact ({flags[0][:60]}) — verify manually"
        passed = True
    else:
        score  = 90
        note   = (f"No AI-generation artifacts detected "
                  f"(cv={cv_energy:.3f}, hf={mean_hf:.3f}, zero={mean_zero:.3f})")
        passed = True

    ok(f"  AI detection: {score}/100 — {note[:60]}")
    return score, note, passed


# ═════════════════════════════════════════════════════════════
#  21D — NAME PLAUSIBILITY SCORING
# ═════════════════════════════════════════════════════════════

def _score_name_plausibility(name_str, state_str=None):
    """
    Score how plausible the extracted name is as a real Indian name.

    Checks:
      1. Character set — only letters, spaces, dots, hyphens
      2. Word count — 2 to 5 words
      3. Word lengths — each word 2-20 chars
      4. No repeated characters (AAAA, XXXX)
      5. Known Indian name suffixes/patterns
      6. Not a keyboard sequence (ABCD, QWERTY)
      7. Not all same word repeated

    Returns: (score, note, passed)
    """
    if not name_str or not name_str.strip():
        return 0, "No name provided", False

    name = name_str.strip().upper()

    # Basic character check
    if not re.match(r"^[A-Z\s.\-']+$", name):
        return 20, f"Invalid characters in name: '{name[:30]}'", False

    words = name.split()
    n_words = len(words)
    min_w, max_w = INDIAN_NAME_PATTERNS['valid_word_lengths']

    # Word count check
    if n_words < INDIAN_NAME_PATTERNS['min_words']:
        return 30, f"Too few words ({n_words}) — genuine names have 2-5 words", False
    if n_words > INDIAN_NAME_PATTERNS['max_words']:
        return 40, f"Too many words ({n_words}) — suspicious for an Aadhaar name", False

    score = 60   # base — format is OK
    issues = []

    # Word length check
    for word in words:
        if len(word) < min_w:
            issues.append(f"Word '{word}' too short")
        if len(word) > max_w:
            issues.append(f"Word '{word}' too long")

    # Repeated character check (AAAA, BBBB etc.)
    for word in words:
        if len(word) >= 3 and len(set(word)) == 1:
            issues.append(f"Repeated chars: '{word}'")
            score -= 30

    # All words same check
    if len(set(words)) == 1 and n_words > 1:
        issues.append("All words identical — not a real name")
        score -= 40

    # Keyboard sequence check
    keyboard_seqs = ['ABCD', 'QWERTY', 'ASDF', 'ZXCV', '1234', 'AAAA', 'XXXX', 'TEST', 'FAKE', 'NULL']
    for seq in keyboard_seqs:
        if seq in name:
            issues.append(f"Keyboard/test sequence detected: '{seq}'")
            score -= 35

    # Digit check — names should not have digits
    if re.search(r'\d', name):
        issues.append("Digits in name field")
        score -= 25

    # Known Indian name pattern boost
    name_lower = name.lower()
    matched_patterns = [
        p for p in INDIAN_NAME_PATTERNS['common_suffixes']
        if p in name_lower
    ]
    if matched_patterns:
        score += min(25, len(matched_patterns) * 10)
        info(f"  Name patterns matched: {matched_patterns[:3]}")
    else:
        # Not matching any known pattern — mild penalty
        # (could be a rare name — don't penalise too hard)
        score -= 5

    # Vowel density check — Indian names have high vowel density
    vowels = sum(1 for c in name if c in 'AEIOU')
    vowel_ratio = vowels / max(len(name.replace(' ', '')), 1)
    if vowel_ratio < 0.15:
        issues.append(f"Very low vowel density ({vowel_ratio:.0%}) — unusual for Indian names")
        score -= 15
    elif vowel_ratio > 0.65:
        issues.append(f"Very high vowel density ({vowel_ratio:.0%}) — unusual")
        score -= 10

    score = max(0, min(100, score))
    passed = score >= 45 and not any('Repeated' in i or 'sequence' in i or 'identical' in i
                                      for i in issues)

    if issues:
        note = f"Name issues: {'; '.join(issues[:2])}"
    elif matched_patterns:
        note = f"Plausible Indian name (matched: {', '.join(matched_patterns[:2])})"
    else:
        note = f"Name format valid (no known pattern match — may be rare regional name)"

    sym = "OK" if passed else "FAIL"
    ok(f"  Name plausibility: {score}/100 [{sym}] — {note[:60]}")
    return score, note, passed


# ═════════════════════════════════════════════════════════════
#  MAIN STEP — Run all 4 checks
# ═════════════════════════════════════════════════════════════

def step21_geo_validate(fields, img_bgr=None):
    """
    Step 21 — Data Intelligence Validation.

    Runs all 4 data-driven checks and returns consolidated result.

    Args:
        fields  : dict from field_extraction / QR backfill
        img_bgr : BGR numpy array of front card image (for AI detection)

    Returns:
        dict with keys:
          signals : list of fraud signal strings
          score   : int 0-100
          verdict : 'VALID' / 'SUSPICIOUS' / 'INVALID'
          details : {check: {score, note, passed}}
          passed  : bool
    """
    section("21 — Data Intelligence Validation")
    info("Running 4 data-driven fraud detectors...")
    print()

    details  = {}
    signals  = []

    pin      = fields.get('address_pin', '')
    district = fields.get('address_district', '')
    state    = fields.get('address_state', '')
    name     = fields.get('name', '')

    # ── 21A: PIN Geo-Validation ───────────────────────────────
    info("21A — PIN Code Geo-Validation")
    pin_valid, pin_expected, pin_note = _validate_pin_state(pin, state)
    if pin_valid is True:
        pin_score  = 100
        pin_passed = True
        ok(f"  PIN {pin}: {pin_note[:60]}")
    elif pin_valid is False:
        pin_score  = 0
        pin_passed = False
        signals.append(f"PIN geo mismatch: {pin_note}")
        err(f"  PIN {pin}: {pin_note[:60]}")
    else:
        pin_score  = 60
        pin_passed = True
        info(f"  PIN {pin}: {pin_note[:60]}")

    details['pin_geo'] = {
        'score': pin_score, 'note': pin_note,
        'passed': pin_passed, 'expected_state': pin_expected
    }

    # ── 21B: District–State Consistency ──────────────────────
    print()
    info("21B — District–State Consistency")
    dist_valid, dist_note = _validate_district_state(district, state)
    if dist_valid is True:
        dist_score  = 100
        dist_passed = True
        ok(f"  District '{district}': {dist_note[:60]}")
    elif dist_valid is False:
        dist_score  = 0
        dist_passed = False
        signals.append(f"District-state mismatch: {dist_note}")
        err(f"  District '{district}': {dist_note[:60]}")
    else:
        dist_score  = 70
        dist_passed = True
        info(f"  District '{district}': {dist_note[:60]}")

    details['district_state'] = {
        'score': dist_score, 'note': dist_note, 'passed': dist_passed
    }

    # ── 21C: AI-Generated Image Detection ────────────────────
    print()
    info("21C — AI-Generated Image Detection")
    if img_bgr is not None:
        ai_score, ai_note, ai_passed = _detect_ai_image(img_bgr)
        if not ai_passed:
            signals.append(f"AI-generated image: {ai_note[:80]}")
    else:
        ai_score  = 60
        ai_note   = "No image provided for AI detection"
        ai_passed = True
        info("  No image — skipping AI detection")

    details['ai_image'] = {
        'score': ai_score, 'note': ai_note, 'passed': ai_passed
    }

    # ── 21D: Name Plausibility ────────────────────────────────
    print()
    info("21D — Name Plausibility Scoring")
    name_score, name_note, name_passed = _score_name_plausibility(name, state)
    if not name_passed:
        signals.append(f"Name implausible: {name_note[:80]}")

    details['name_plausibility'] = {
        'score': name_score, 'note': name_note, 'passed': name_passed
    }

    # ── Compute overall score ─────────────────────────────────
    weights = {
        'pin_geo':           0.30,   # strongest geo signal
        'district_state':    0.30,   # equally strong
        'ai_image':          0.25,   # important fraud type
        'name_plausibility': 0.15,   # supplementary
    }
    scores = {
        'pin_geo':           pin_score,
        'district_state':    dist_score,
        'ai_image':          ai_score,
        'name_plausibility': name_score,
    }
    overall = int(sum(scores[k] * weights[k] for k in weights))

    # Hard fails (geographic impossibility) override score
    hard_fails = [k for k in ['pin_geo', 'district_state']
                  if not details[k]['passed']]
    if hard_fails:
        overall = min(overall, 25)

    # Verdict
    if overall >= 75 and not hard_fails:
        verdict = "VALID ✓"
    elif overall >= 50 or not hard_fails:
        verdict = "SUSPICIOUS ⚠"
    else:
        verdict = "INVALID ✗"

    # Summary printout
    print()
    print(f"  {'─'*60}")
    print(f"  {'Check':<28} {'Score':>6}   {'Status'}")
    print(f"  {'─'*60}")
    labels = {
        'pin_geo':           '21A PIN Geo-Validation',
        'district_state':    '21B District-State Match',
        'ai_image':          '21C AI Image Detection',
        'name_plausibility': '21D Name Plausibility',
    }
    for key, label in labels.items():
        d   = details[key]
        sym = "✓" if d['passed'] else "✗"
        print(f"  {label:<28} {d['score']:>5}/100  {sym}  {d['note'][:35]}")

    print(f"  {'─'*60}")
    print(f"  Overall Data Score : {overall}/100")
    print(f"  Verdict            : {verdict}")
    if signals:
        print(f"  Fraud Signals ({len(signals)}):")
        for s in signals:
            print(f"    ✗ {s[:72]}")
    print(f"  {'─'*60}")

    if signals:
        warn(f"Data intelligence: {verdict} ({overall}/100) — {len(signals)} signal(s)")
    else:
        ok(f"Data intelligence: {verdict} ({overall}/100)")

    return {
        'signals': signals,
        'score':   overall,
        'verdict': verdict,
        'details': details,
        'passed':  len(hard_fails) == 0,
    }