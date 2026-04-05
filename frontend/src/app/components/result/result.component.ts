import {
  Component, Input, Output, EventEmitter,
  ChangeDetectionStrategy, OnInit, inject
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { VerificationResponse } from '../../models/verification.model';
import { VerificationTableComponent } from '../verification-table/verification-table.component';
import { ScoreCardComponent, ScoreCardData } from '../score-card/score-card.component';
import { FraudSignalsComponent } from '../fraud-signals/fraud-signals.component';
import { VerificationService } from '../../services/verification.service';

type ResultTab = 'overview' | 'fields' | 'modules' | 'signals';

@Component({
  selector: 'app-result',
  standalone: true,
  imports: [
    CommonModule,
    VerificationTableComponent,
    ScoreCardComponent,
    FraudSignalsComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './result.component.html',
  styleUrls: ['./result.component.scss'],
})
export class ResultComponent implements OnInit {
  @Input({ required: true }) result!: VerificationResponse;
  @Output() reset = new EventEmitter<void>();

  private svc = inject(VerificationService);

  activeTab: ResultTab = 'overview';

  get verdictClass(): string {
    const l = this.result.verdict.label.toUpperCase();
    if (l.includes('VERIFIED') || l.includes('GENUINE') || l.includes('VALID') && !l.includes('INVALID'))
      return 'verdict-verified';
    if (l.includes('REVIEW') || l.includes('SUSPICIOUS') || l.includes('LIKELY'))
      return 'verdict-review';
    return 'verdict-rejected';
  }

  get verdictIcon(): string {
    const l = this.result.verdict.label.toUpperCase();
    if (l.includes('VERIFIED') || l.includes('GENUINE') || (l.includes('VALID') && !l.includes('INVALID')))
      return '✓';
    if (l.includes('REVIEW') || l.includes('SUSPICIOUS') || l.includes('LIKELY'))
      return '⚠';
    return '✗';
  }

  get overallScore(): number {
    const qr = this.result.qr?.trust_score ?? 0;
    const t  = this.result.tamper?.score   ?? 100;
    const g  = this.result.geo?.score      ?? 100;
    const f  = this.result.face?.quality_score ?? 0;
    return Math.round(qr * 0.35 + t * 0.25 + g * 0.25 + f * 0.15);
  }

  get ringColor(): string {
    const s = this.overallScore;
    if (s >= 75) return 'var(--green)';
    if (s >= 50) return 'var(--yellow)';
    return 'var(--red)';
  }

  get ringDashOffset(): number {
    return 226 - (226 * this.overallScore / 100);
  }

  get scoreCards(): ScoreCardData[] {
    const r = this.result;
    const toDetails = (obj: Record<string, any> | undefined) =>
      obj ? Object.entries(obj).map(([k, v]) => ({
        label: k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        score: v.score ?? 0,
        note:  v.note  ?? '',
        passed: v.passed ?? true,
      })) : [];

    return [
      {
        label:   'QR Verification',
        score:   r.qr?.trust_score ?? 0,
        verdict: r.qr?.verdict ?? 'N/A',
        passed:  (r.qr?.trust_score ?? 0) >= 60,
        icon:    '📱',
      },
      {
        label:   'Anti-Tampering',
        score:   r.tamper?.score ?? 0,
        verdict: r.tamper?.verdict ?? 'N/A',
        passed:  r.tamper?.passed ?? true,
        icon:    '🛡',
        details: toDetails(r.tamper?.details as any),
      },
      {
        label:   'Geo-Validation',
        score:   r.geo?.score ?? 0,
        verdict: r.geo?.verdict ?? 'N/A',
        passed:  r.geo?.passed ?? true,
        icon:    '📍',
        details: toDetails(r.geo?.details as any),
      },
      {
        label:   'Face Quality',
        score:   r.face?.quality_score ?? 0,
        verdict: r.face?.quality_verdict ?? 'N/A',
        passed:  (r.face?.quality_score ?? 0) >= 50,
        icon:    '👁',
      },
      {
        label:   'Liveness',
        score:   Math.round(r.face?.combined_liveness_score ?? r.face?.liveness_score ?? 0),
        verdict: (r.face?.combined_liveness_real ?? r.face?.likely_real)
                 ? 'Real face' : 'Suspicious',
        passed:  r.face?.combined_liveness_real ?? r.face?.likely_real ?? true,
        icon:    '🔬',
      },
    ];
  }

  trackByScoreCard(_index: number, card: ScoreCardData): string {
    return card.label;
  }

  get totalSignals(): number {
    return this.result.explanations?.length ?? 0;
  }

  get criticalSignals(): number {
    return this.result.explanations?.filter(e => e.severity === 'critical').length ?? 0;
  }

  // Resolve a field: use OCR value, fall back to QR field
  private resolve(ocrVal: string | null | undefined, qrKey?: string): string {
    if (ocrVal && ocrVal.trim()) return ocrVal;
    if (qrKey) {
      const qf = this.result.qr?.qr_fields as any;
      const v  = qf?.[qrKey] ?? qf?.[qrKey.replace('address_','')] ?? '';
      if (v && v.trim()) return v;
    }
    return '';
  }

  get identityQuickFields() {
    const f  = this.result.fields;
    const qf = this.result.qr?.qr_fields as any ?? {};
    return [
      { label: 'Name',         value: this.resolve(f.name, 'name')           || qf.name   || '—' },
      { label: 'Date of Birth',value: this.resolve(f.dob, 'dob')             || qf.dob    || '—' },
      { label: 'Gender',       value: this.resolve(f.gender, 'gender')        || qf.gender || '—' },
      { label: 'Aadhaar No.',  value: this.maskAadhaar(f.aadhaar_number)              || '—' },
      { label: 'Card Type',    value: f.card_type                                      || '—' },
    ];
  }

  get addressQuickFields() {
    const f  = this.result.fields;
    const qf = this.result.qr?.qr_fields as any ?? {};
    // OCR with QR fallback for each address component
    const house    = f.address_house      || qf.building || '';
    const locality = f.address_locality   || qf.locality || '';
    const district = f.address_district   || qf.district || '';
    const state    = f.address_state      || qf.state    || '';
    const pin      = f.address_pin        || qf.pin      || '';
    const parts    = [house, locality, district, state, pin].filter(Boolean);
    return parts.join(', ') || '—';
  }

  get ageInfo(): string {
    const d = this.result.face?.age_detail;
    if (!d) return '';
    return `Estimated ~${Math.round(d.estimated_age)} yrs · Declared ${Math.round(d.declared_age)} yrs`;
  }

  maskAadhaar(n: string | null): string {
    if (!n) return '—';
    const c = n.replace(/\s/g, '');
    return c.length === 12 ? `XXXX XXXX ${c.slice(8)}` : n;
  }

  scoreColor(s: number): string {
    if (s >= 75) return 'var(--green)';
    if (s >= 50) return 'var(--yellow)';
    return 'var(--red)';
  }

  downloadPDF(): void {
    if (this.result.pdf_report_url) {
      window.open(this.result.pdf_report_url, '_blank');
    }
  }

  ngOnInit(): void {
    // Scroll to top of result
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}
