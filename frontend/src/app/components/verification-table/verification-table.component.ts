import { Component, Input, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { VerificationFields, QRResult } from '../../models/verification.model';

interface TableRow {
  label:    string;
  ocr:      string;
  qr:       string;
  match:    boolean | null;
  note:     string;
}

@Component({
  selector: 'app-verification-table',
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="table-wrap">
      <!-- Identity -->
      <div class="section-label">Identity Fields</div>
      <table class="data-table">
        <thead>
          <tr>
            <th class="col-field">Field</th>
            <th class="col-ocr">OCR (image)</th>
            <th class="col-qr">QR (code)</th>
            <th class="col-status">Status</th>
          </tr>
        </thead>
        <tbody>
          <tr *ngFor="let row of identityRows">
            <td class="label-col">{{ row.label }}</td>
            <td class="mono-col">{{ row.ocr || '—' }}</td>
            <td class="mono-col">{{ row.qr || '—' }}</td>
            <td class="status-col">
              <span class="status-badge"
                    [class.s-pass]="row.match === true"
                    [class.s-fail]="row.match === false"
                    [class.s-na]="row.match === null">
                <span class="s-dot"></span>
                {{ row.match === true ? 'Match ✓' : row.match === false ? 'Mismatch ✗' : (row.qr && row.qr !== '—' ? 'QR found' : 'OCR only') }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>

      <!-- Address -->
      <div class="section-label" style="margin-top:20px">Address Fields</div>
      <table class="data-table">
        <thead>
          <tr>
            <th class="col-field">Field</th>
            <th class="col-ocr">OCR (image)</th>
            <th class="col-qr">QR (code)</th>
            <th class="col-status">Status</th>
          </tr>
        </thead>
        <tbody>
          <tr *ngFor="let row of addressRows">
            <td class="label-col">{{ row.label }}</td>
            <td class="mono-col">{{ row.ocr || '—' }}</td>
            <td class="mono-col">{{ row.qr || '—' }}</td>
            <td class="status-col">
              <span class="status-badge"
                    [class.s-pass]="row.match === true"
                    [class.s-fail]="row.match === false"
                    [class.s-na]="row.match === null">
                <span class="s-dot"></span>
                {{ row.match === true ? 'Match ✓' : row.match === false ? 'Mismatch ✗' : (row.qr && row.qr !== '—' ? 'QR found' : 'Info') }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>

      <!-- QR metadata -->
      <div class="qr-meta" *ngIf="qr.qr_decoded">
        <span class="mono" style="color:var(--text-tertiary);font-size:0.72rem">
          QR Format: {{ qr.qr_format || '—' }} ·
          Trust Score: {{ qr.trust_score }}/100 ·
          {{ qr.qr_decoded ? 'Decoded ✓' : 'Not decoded' }}
          <span *ngIf="qr.qr_fields?.mobile_masked"> · Mobile: {{ qr.qr_fields.mobile_masked }}</span>
        </span>
      </div>
    </div>
  `,
  styles: [`
    .table-wrap { overflow-x: auto; }

    .col-field  { width: 130px; }
    .col-ocr    { width: 35%; }
    .col-qr     { width: 35%; }
    .col-status { width: 110px; }

    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 0.72rem;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 20px;
      white-space: nowrap;
    }

    .s-dot {
      width: 5px; height: 5px;
      border-radius: 50%;
      flex-shrink: 0;
    }

    .s-pass {
      background: var(--green-dim);
      color: var(--green);
      border: 1px solid var(--green-border);
      .s-dot { background: var(--green); }
    }
    .s-fail {
      background: var(--red-dim);
      color: var(--red);
      border: 1px solid var(--red-border);
      .s-dot { background: var(--red); }
    }
    .s-na {
      background: var(--bg-elevated);
      color: var(--text-tertiary);
      border: 1px solid var(--border-dim);
      .s-dot { background: var(--text-tertiary); }
    }

    .qr-meta {
      margin-top: 10px;
      padding: 8px 14px;
      background: var(--bg-elevated);
      border-radius: var(--r-sm);
    }

    .data-table { font-size: 0.83rem; }
    .mono-col { font-size: 0.78rem; }
  `],
})
export class VerificationTableComponent {
  @Input({ required: true }) fields!: VerificationFields;
  @Input({ required: true }) qr!: QRResult;

  private mask(n: string | null): string {
    if (!n) return '';
    const c = n.replace(/\s/g, '');
    return c.length === 12 ? `XXXX XXXX ${c.slice(8)}` : n;
  }

  private fc(key: string): boolean | null {
    return this.qr?.field_checks?.[key]?.match ?? null;
  }

  private qf(key: string): string {
    const qf = this.qr?.qr_fields as any;
    if (!qf) return '';
    // Try the key directly, then with address_ prefix stripped
    return qf[key] ?? qf[key.replace('address_','')] ?? '';
  }

  get identityRows(): TableRow[] {
    return [
      { label: 'Name',         ocr: this.fields.name ?? '',
        qr: this.qf('name'),   match: this.fc('name'),   note: '' },
      { label: 'Date of Birth',ocr: this.fields.dob ?? '',
        qr: this.qf('dob'),    match: this.fc('dob'),    note: '' },
      { label: 'Gender',       ocr: this.fields.gender ?? '',
        qr: this.qf('gender'), match: this.fc('gender'), note: '' },
      { label: 'Aadhaar No.', ocr: this.mask(this.fields.aadhaar_number),
        qr: '— (not in Secure QR)', match: null, note: '' },
      { label: 'Virtual ID',  ocr: this.fields.vid ?? '—',
        qr: '—',               match: null,              note: '' },
    ];
  }

  // Helper: show OCR value, fall back to QR value if OCR is empty
  private ocr(val: string | null | undefined, qrKey?: string): string {
    if (val && val.trim()) return val;
    if (qrKey) return this.qf(qrKey);
    return '';
  }

  get addressRows(): TableRow[] {
    const f = this.fields;
    return [
      { label: 'PIN Code',
        ocr:   this.ocr(f.address_pin, 'pin'),
        qr:    this.qf('pin'),
        match: this.fc('pin'), note: '' },
      { label: 'District',
        ocr:   this.ocr(f.address_district, 'district'),
        qr:    this.qf('district'),
        match: this.fc('district'), note: '' },
      { label: 'State',
        ocr:   this.ocr(f.address_state, 'state'),
        qr:    this.qf('state'),
        match: this.fc('state'), note: '' },
      { label: 'Sub-district',
        ocr:   this.ocr(f.address_subdistrict, 'vtc'),
        qr:    this.qf('vtc'),
        match: null, note: '' },
      { label: 'Locality',
        ocr:   this.ocr(f.address_locality, 'locality'),
        qr:    this.qf('locality'),
        match: null, note: '' },
      { label: 'Landmark',
        ocr:   this.ocr(f.address_landmark, 'landmark'),
        qr:    this.qf('landmark'),
        match: null, note: '' },
      { label: 'House / Bldg',
        ocr:   this.ocr(f.address_house, 'building'),
        qr:    this.qf('building'),
        match: null, note: '' },
    ];
  }
}
