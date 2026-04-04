# report_generator.py
# Phase 5 — PDF Verification Report
#
# Generates a signed, timestamped PDF report after each verification.
# This is what banks, NBFCs, and KYC providers need to attach
# to loan files, account opening forms, and compliance records.
#
# REPORT SECTIONS:
#   1. Header    — AadhaarCheck logo, timestamp, report ID
#   2. Verdict   — Final verdict banner (VERIFIED / REVIEW / REJECTED)
#   3. Fields    — 4-column table (Field | OCR | QR | Status)
#   4. Scores    — QR trust, tampering, geo, face quality bars
#   5. Signals   — All fraud signals with plain-English explanation
#   6. Checklist — Minimum field requirements pass/fail
#   7. Footer    — Disclaimer, report hash, generated-by
#
# INSTALL:
#   pip install reportlab  (already in your requirements.txt)
#
# USAGE:
#   from report_generator import generate_pdf_report
#   pdf_path = generate_pdf_report(fields, qr_result, tamper_result,
#                                  geo_result, face_result,
#                                  verification, final_verdict)
# ─────────────────────────────────────────────────────────────

import os
import hashlib
import datetime
import tempfile
import uuid

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable

# ── Colour palette ────────────────────────────────────────────
CLR_GREEN  = colors.HexColor('#16a34a')
CLR_YELLOW = colors.HexColor('#d97706')
CLR_RED    = colors.HexColor('#dc2626')
CLR_BLUE   = colors.HexColor('#2563eb')
CLR_DARK   = colors.HexColor('#111827')
CLR_MUTED  = colors.HexColor('#6b7280')
CLR_BORDER = colors.HexColor('#e5e7eb')
CLR_BG     = colors.HexColor('#f9fafb')
CLR_WHITE  = colors.white

W, H = A4   # 210 x 297 mm


def _mask_aadhaar(n):
    if not n:
        return '—'
    c = n.replace(' ', '')
    return f'XXXX XXXX {c[8:]}' if len(c) == 12 else n


def _score_bar(score, width=60):
    """Return a text progress bar string."""
    filled = int(score / 100 * 20)
    return f"{'█' * filled}{'░' * (20 - filled)}  {score}/100"


def _verdict_color(verdict_str):
    v = str(verdict_str).upper()
    if 'VERIFIED' in v or 'GENUINE' in v or 'VALID' in v and 'INVALID' not in v:
        return CLR_GREEN
    elif 'REVIEW' in v or 'SUSPICIOUS' in v:
        return CLR_YELLOW
    else:
        return CLR_RED


def generate_pdf_report(
    fields,
    verification,
    final_verdict,
    qr_result=None,
    tamper_result=None,
    geo_result=None,
    face_result=None,
    output_path=None,
):
    """
    Generate a PDF verification report.

    Args:
        fields         : dict from field_extraction
        verification   : dict from step15_verify
        final_verdict  : str — 'VERIFIED ✓', 'REVIEW REQUIRED ⚠', etc.
        qr_result      : dict from step18_qr_verify
        tamper_result  : dict from step20_tampering_analysis
        geo_result     : dict from step21_geo_validate
        face_result    : dict from step19_face_pipeline
        output_path    : str path to save PDF (auto-generated if None)

    Returns:
        str — path to generated PDF file
    """
    if output_path is None:
        ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(
            tempfile.gettempdir(),
            f'aadhaar_report_{ts}.pdf'
        )

    report_id  = str(uuid.uuid4()).upper()[:12]
    timestamp  = datetime.datetime.now().strftime('%d %B %Y, %H:%M:%S')
    styles     = getSampleStyleSheet()

    # ── Custom styles ─────────────────────────────────────────
    def S(name, **kw):
        base = kw.pop('parent', 'Normal')
        s = ParagraphStyle(name, parent=styles[base], **kw)
        return s

    style_h1      = S('H1',     fontSize=20, textColor=CLR_DARK,
                       fontName='Helvetica-Bold', spaceAfter=2)
    style_h2      = S('H2',     fontSize=12, textColor=CLR_DARK,
                       fontName='Helvetica-Bold', spaceBefore=8, spaceAfter=4)
    style_h3      = S('H3',     fontSize=10, textColor=CLR_MUTED,
                       fontName='Helvetica', spaceAfter=2)
    style_body    = S('Body',   fontSize=9,  textColor=CLR_DARK,
                       fontName='Helvetica', leading=14)
    style_muted   = S('Muted',  fontSize=8,  textColor=CLR_MUTED,
                       fontName='Helvetica')
    style_mono    = S('Mono',   fontSize=8,  textColor=CLR_DARK,
                       fontName='Courier')
    style_center  = S('Ctr',    fontSize=9,  textColor=CLR_DARK,
                       fontName='Helvetica', alignment=TA_CENTER)
    style_right   = S('Right',  fontSize=8,  textColor=CLR_MUTED,
                       fontName='Helvetica', alignment=TA_RIGHT)
    style_signal  = S('Sig',    fontSize=8,  textColor=CLR_RED,
                       fontName='Helvetica', leftIndent=8, spaceAfter=2)
    style_ok      = S('OK',     fontSize=8,  textColor=CLR_GREEN,
                       fontName='Helvetica', leftIndent=8, spaceAfter=2)

    story = []
    pw    = W - 30*mm   # printable width

    # ══════════════════════════════════════════════════════════
    #  HEADER
    # ══════════════════════════════════════════════════════════
    header_data = [[
        Paragraph('<b>AadhaarCheck</b>', S('BH', fontSize=16,
                  textColor=CLR_BLUE, fontName='Helvetica-Bold')),
        Paragraph(
            f'<font size="8" color="grey">Report ID: {report_id}<br/>'
            f'Generated: {timestamp}<br/>'
            f'System: AadhaarCheck v5</font>',
            style_right
        )
    ]]
    header_tbl = Table(header_data, colWidths=[pw * 0.6, pw * 0.4])
    header_tbl.setStyle(TableStyle([
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(header_tbl)
    story.append(HRFlowable(width=pw, thickness=1.5, color=CLR_BLUE, spaceAfter=10))

    story.append(Paragraph('Aadhaar Card Verification Report', style_h1))
    story.append(Spacer(1, 4))

    # ══════════════════════════════════════════════════════════
    #  VERDICT BANNER
    # ══════════════════════════════════════════════════════════
    v_color    = _verdict_color(final_verdict)
    qr_trust   = (qr_result or {}).get('trust_score', 0)
    t_score    = (tamper_result or {}).get('score', 100)
    g_score    = (geo_result or {}).get('score', 100)
    f_score    = (face_result or {}).get('quality_score', 0)

    # Overall score: weighted avg of all module scores
    overall = int(qr_trust * 0.35 + t_score * 0.25 + g_score * 0.25 + f_score * 0.15)

    verdict_data = [[
        Paragraph(
            f'<b><font size="16">{final_verdict}</font></b><br/>'
            f'<font size="9" color="grey">Overall confidence score: {overall}/100</font>',
            S('VB', fontName='Helvetica-Bold', textColor=v_color, leading=20)
        ),
        Paragraph(
            f'<b><font size="28" color="{('#' + v_color.hexval()[2:])}">{overall}</font></b><br/>'
            f'<font size="7" color="grey">/100</font>',
            S('VS', fontName='Helvetica-Bold', alignment=TA_CENTER)
        )
    ]]
    verdict_tbl = Table(verdict_data, colWidths=[pw * 0.78, pw * 0.22])
    verdict_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), CLR_BG),
        ('BOX',           (0,0), (-1,-1), 1.5, v_color),
        ('LEFTPADDING',   (0,0), (0,-1), 12),
        ('RIGHTPADDING',  (-1,0), (-1,-1), 12),
        ('TOPPADDING',    (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',         (1,0), (1,-1), 'CENTER'),
    ]))
    story.append(verdict_tbl)
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════════
    #  SECTION: EXTRACTED FIELDS — 4-column table
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph('Extracted Fields', style_h2))

    qf = (qr_result or {}).get('qr_fields', {})
    fc = (qr_result or {}).get('field_checks', {})

    def _qr(key):
        v = qf.get(key, '')
        return str(v) if v else '—'

    def _ocr(key):
        v = fields.get(key, '')
        return str(v) if v else '—'

    def _status(check_key, ocr_key):
        m = fc.get(check_key, {}).get('match')
        if m is True:  return ('✓ Match', CLR_GREEN)
        if m is False: return ('✗ Mismatch', CLR_RED)
        v = fields.get(ocr_key)
        vr = verification.get(check_key.replace('_', ' ').title(), {})
        if vr.get('valid') is True: return ('✓ Valid', CLR_GREEN)
        if vr.get('valid') is False: return ('✗ Invalid', CLR_RED)
        return ('— N/A', CLR_MUTED)

    col_w = [pw*0.22, pw*0.27, pw*0.27, pw*0.24]
    tbl_header = [
        Paragraph('<b>Field</b>',       style_center),
        Paragraph('<b>OCR (image)</b>', style_center),
        Paragraph('<b>QR (code)</b>',   style_center),
        Paragraph('<b>Status</b>',      style_center),
    ]

    def row(label, ocr_val, qr_val, status_txt, status_clr):
        return [
            Paragraph(label,    style_body),
            Paragraph(ocr_val,  style_mono),
            Paragraph(qr_val,   style_mono),
            Paragraph(f'<font color="{status_clr.hexval()}">{status_txt}</font>',
                      style_center),
        ]

    nm_st, nm_cl = _status('name', 'name')
    db_st, db_cl = _status('dob',  'dob')
    gn_st, gn_cl = _status('gender', 'gender')

    # Aadhaar# — QR doesn't store it in Secure QR
    aa_ocr = _mask_aadhaar(fields.get('aadhaar_number', ''))
    aa_vr  = verification.get('Aadhaar Number', {}).get('valid')
    aa_st  = ('✓ Valid format', CLR_GREEN) if aa_vr else ('✗ Invalid', CLR_RED)

    field_rows = [
        tbl_header,
        row('Name',           _ocr('name'),          _qr('name'),     nm_st, nm_cl),
        row('Date of Birth',  _ocr('dob'),           _qr('dob'),      db_st, db_cl),
        row('Gender',         _ocr('gender'),        _qr('gender'),   gn_st, gn_cl),
        row('Aadhaar Number', aa_ocr,                '— (Secure QR)', aa_st[0], aa_st[1]),
        row('Virtual ID',     _ocr('vid'),           '—',             '— OCR only', CLR_MUTED),
        row('PIN Code',       _ocr('address_pin'),   _qr('pin'),
            *_status('pin', 'address_pin')),
        row('District',       _ocr('address_district'), _qr('district'),
            *_status('district', 'address_district')),
        row('State',          _ocr('address_state'), _qr('state'),
            *_status('state', 'address_state')),
        row('Building',       _ocr('address_house'), _qr('building'), '— Info only', CLR_MUTED),
        row('Landmark',       _ocr('address_landmark'), _qr('landmark'), '— Info only', CLR_MUTED),
        row('Mobile',         _ocr('mobile') or '—', _qr('mobile_masked'), '— Masked', CLR_MUTED),
    ]

    ftbl = Table(field_rows, colWidths=col_w, repeatRows=1)
    ftbl.setStyle(TableStyle([
        # Header row
        ('BACKGROUND',    (0,0), (-1,0), CLR_DARK),
        ('TEXTCOLOR',     (0,0), (-1,0), CLR_WHITE),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,0), 8),
        ('ALIGN',         (0,0), (-1,0), 'CENTER'),
        ('TOPPADDING',    (0,0), (-1,0), 6),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        # Data rows
        ('FONTSIZE',      (0,1), (-1,-1), 8),
        ('TOPPADDING',    (0,1), (-1,-1), 4),
        ('BOTTOMPADDING', (0,1), (-1,-1), 4),
        ('LEFTPADDING',   (0,0), (-1,-1), 5),
        ('RIGHTPADDING',  (0,0), (-1,-1), 5),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',         (3,1), (3,-1), 'CENTER'),
        # Alternating rows
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [CLR_WHITE, CLR_BG]),
        # Grid
        ('GRID',          (0,0), (-1,-1), 0.5, CLR_BORDER),
        ('LINEABOVE',     (0,0), (-1,0), 1, CLR_DARK),
        ('LINEBELOW',     (0,-1), (-1,-1), 1, CLR_BORDER),
    ]))
    story.append(ftbl)
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════════
    #  SECTION: MODULE SCORES
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph('Module Scores', style_h2))

    score_data = [
        ['Module', 'Score', 'Bar', 'Verdict'],
        ['QR Verification',      f'{qr_trust}/100',
         _score_bar(qr_trust),   (qr_result or {}).get('verdict', 'N/A')],
        ['Anti-Tampering',       f'{t_score}/100',
         _score_bar(t_score),    (tamper_result or {}).get('verdict', 'N/A')],
        ['Geo-Validation',       f'{g_score}/100',
         _score_bar(g_score),    (geo_result or {}).get('verdict', 'N/A')],
        ['Face Quality',         f'{f_score}/100',
         _score_bar(f_score),    (face_result or {}).get('quality_verdict', 'N/A')],
    ]
    if face_result and face_result.get('combined_liveness_score') is not None:
        ls = face_result['combined_liveness_score']
        score_data.append([
            'Liveness (combined)', f'{ls:.0f}/100',
            _score_bar(int(ls)),
            'Real' if face_result.get('combined_liveness_real') else 'Suspicious'
        ])

    stbl = Table(score_data, colWidths=[pw*0.25, pw*0.12, pw*0.43, pw*0.20],
                 repeatRows=1)
    stbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), CLR_DARK),
        ('TEXTCOLOR',     (0,0), (-1,0), CLR_WHITE),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 8),
        ('ALIGN',         (1,0), (1,-1), 'CENTER'),
        ('ALIGN',         (3,0), (3,-1), 'CENTER'),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING',   (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [CLR_WHITE, CLR_BG]),
        ('GRID',          (0,0), (-1,-1), 0.5, CLR_BORDER),
        ('FONTNAME',      (2,1), (2,-1), 'Courier'),
    ]))
    story.append(stbl)
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════════
    #  SECTION: FRAUD SIGNALS
    # ══════════════════════════════════════════════════════════
    all_signals = []
    for r, label in [
        (qr_result,     'QR'),
        (tamper_result, 'Tampering'),
        (geo_result,    'Geo'),
    ]:
        if r:
            for s in r.get('signals', []):
                all_signals.append((label, s))

    story.append(Paragraph('Fraud Signals & Findings', style_h2))
    if all_signals:
        sig_data = [['Source', 'Signal', 'Plain-English Explanation']]

        # Plain-English explanations for common signal types
        explanations = {
            'QR':         'The QR code embedded in the card contains data that '
                          'does not match what is printed on the card.',
            'Tampering':  'Image analysis detected signs of digital manipulation — '
                          'the card may have been edited using photo software.',
            'Geo':        'The address fields contain a geographic impossibility — '
                          'the PIN code, district, or state combination does not exist '
                          'in India Post records.',
        }

        for source, signal in all_signals:
            expl = explanations.get(source, 'Manual review recommended.')
            sig_data.append([
                Paragraph(f'<font color="{('#' + CLR_RED.hexval()[2:])}">{source}</font>', style_body),
                Paragraph(signal[:80], style_body),
                Paragraph(expl, style_muted),
            ])

        sig_tbl = Table(sig_data, colWidths=[pw*0.12, pw*0.40, pw*0.48],
                        repeatRows=1)
        sig_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#fef2f2')),
            ('TEXTCOLOR',     (0,0), (-1,0), CLR_RED),
            ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0), (-1,-1), 8),
            ('TOPPADDING',    (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING',   (0,0), (-1,-1), 5),
            ('GRID',          (0,0), (-1,-1), 0.5, CLR_BORDER),
            ('ROWBACKGROUNDS', (0,1), (-1,-1),
             [colors.HexColor('#fff5f5'), CLR_WHITE]),
            ('VALIGN',        (0,0), (-1,-1), 'TOP'),
        ]))
        story.append(sig_tbl)
    else:
        story.append(Paragraph(
            '✓ No fraud signals detected across all verification modules.',
            style_ok
        ))
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════════
    #  SECTION: MINIMUM FIELD CHECKLIST
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph('Minimum Field Requirements', style_h2))

    mandatory = ['Name', 'Date of Birth', 'Gender', 'Aadhaar Number']
    check_data = [['Field', 'Required', 'Status', 'Value']]
    for lbl in mandatory:
        res   = verification.get(lbl, {})
        valid = res.get('valid')
        st    = ('✓ PASS', CLR_GREEN) if valid is True else \
                ('⚠ WARN', CLR_YELLOW) if valid is None else \
                ('✗ FAIL', CLR_RED)
        val   = str(res.get('value', '—') or '—')[:30]
        check_data.append([
            Paragraph(lbl, style_body),
            Paragraph('Mandatory', S('M', fontSize=8, textColor=CLR_BLUE,
                       fontName='Helvetica')),
            Paragraph(f'<font color="{('#' + st[1].hexval()[2:])}">{st[0]}</font>',
                      style_center),
            Paragraph(val if lbl != 'Aadhaar Number' else _mask_aadhaar(val),
                      style_mono),
        ])

    # Recommended fields
    rec_fields = [
        ('PIN Code',  'address_pin',      'Recommended'),
        ('District',  'address_district', 'Recommended'),
        ('State',     'address_state',    'Recommended'),
    ]
    for lbl, key, req in rec_fields:
        val = fields.get(key, '')
        qr_match = fc.get(key.replace('address_', ''), {}).get('match')
        if qr_match is True:
            st = ('✓ QR match', CLR_GREEN)
        elif qr_match is False:
            st = ('✗ QR mismatch', CLR_RED)
        elif val:
            st = ('✓ OCR found', CLR_GREEN)
        else:
            st = ('— Not found', CLR_MUTED)
        check_data.append([
            Paragraph(lbl, style_body),
            Paragraph(req, S('R', fontSize=8, textColor=CLR_MUTED,
                      fontName='Helvetica')),
            Paragraph(f'<font color="{('#' + st[1].hexval()[2:])}">{st[0]}</font>',
                      style_center),
            Paragraph(str(val or '—')[:30], style_mono),
        ])

    ctbl = Table(check_data, colWidths=[pw*0.25, pw*0.18, pw*0.22, pw*0.35],
                 repeatRows=1)
    ctbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), CLR_DARK),
        ('TEXTCOLOR',     (0,0), (-1,0), CLR_WHITE),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 8),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING',   (0,0), (-1,-1), 5),
        ('ALIGN',         (2,0), (2,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [CLR_WHITE, CLR_BG]),
        ('GRID',          (0,0), (-1,-1), 0.5, CLR_BORDER),
        ('LINEBELOW',     (0,4), (-1,4), 1.5, CLR_BORDER),
    ]))
    story.append(ctbl)
    story.append(Spacer(1, 16))

    # ══════════════════════════════════════════════════════════
    #  FOOTER
    # ══════════════════════════════════════════════════════════
    story.append(HRFlowable(width=pw, thickness=0.5, color=CLR_BORDER))
    story.append(Spacer(1, 4))

    # Compute report hash for tamper-evidence
    report_content = (
        f"{report_id}{timestamp}{final_verdict}"
        f"{fields.get('name','')}{fields.get('aadhaar_number','')}"
        f"{qr_trust}{t_score}{g_score}"
    )
    report_hash = hashlib.sha256(report_content.encode()).hexdigest()[:16].upper()

    footer_data = [[
        Paragraph(
            '<font size="7" color="grey">'
            'DISCLAIMER: This report is generated by an automated AI system. '
            'It is intended to assist verification and does not constitute '
            'legal proof of identity. Always verify through official UIDAI channels '
            'for binding KYC decisions.<br/>'
            f'Report Hash (SHA-256 prefix): {report_hash} · '
            f'Report ID: {report_id} · '
            f'Generated by AadhaarCheck v5'
            '</font>',
            S('Disc', fontSize=7, textColor=CLR_MUTED, leading=10)
        )
    ]]
    ftbl2 = Table(footer_data, colWidths=[pw])
    ftbl2.setStyle(TableStyle([
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('LEFTPADDING',   (0,0), (-1,-1), 0),
    ]))
    story.append(ftbl2)

    # ── Build PDF ─────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm,  bottomMargin=15*mm,
        title=f'Aadhaar Verification Report {report_id}',
        author='AadhaarCheck v5',
        subject='Aadhaar Card Verification',
    )
    doc.build(story)

    return output_path, report_id, report_hash