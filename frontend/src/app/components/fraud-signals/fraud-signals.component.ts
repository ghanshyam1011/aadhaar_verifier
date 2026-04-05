import { Component, Input, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Explanation } from '../../models/verification.model';

@Component({
  selector: 'app-fraud-signals',
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="signals-wrap">
      <!-- No signals -->
      <div class="no-signals" *ngIf="!explanations?.length">
        <span class="no-sig-icon">✓</span>
        <div>
          <div class="no-sig-title">No fraud signals detected</div>
          <div class="no-sig-sub">All verification modules passed without raising any flags.</div>
        </div>
      </div>

      <!-- Signals list -->
      <div class="signal-item" *ngFor="let e of explanations"
           [class.critical]="e.severity === 'critical'"
           [class.warning]="e.severity === 'warning'">
        <div class="signal-header">
          <span class="sev-badge"
                [class.sev-critical]="e.severity === 'critical'"
                [class.sev-warning]="e.severity === 'warning'">
            {{ e.severity === 'critical' ? '🚨 CRITICAL' : '⚠ WARNING' }}
          </span>
          <span class="signal-source mono">{{ e.source }}</span>
        </div>
        <div class="signal-technical">{{ e.signal }}</div>
        <div class="signal-plain">{{ e.plain }}</div>
      </div>
    </div>
  `,
  styles: [`
    .signals-wrap {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .no-signals {
      display: flex;
      align-items: flex-start;
      gap: 14px;
      padding: 18px 20px;
      background: var(--green-dim);
      border: 1px solid var(--green-border);
      border-radius: var(--r-lg);
    }

    .no-sig-icon {
      font-size: 1.3rem;
      color: var(--green);
      margin-top: 2px;
    }

    .no-sig-title {
      font-weight: 600;
      color: var(--green);
      margin-bottom: 2px;
    }

    .no-sig-sub {
      font-size: 0.82rem;
      color: var(--text-secondary);
    }

    /* Signal card */
    .signal-item {
      border-radius: var(--r-lg);
      padding: 16px 18px;
      border: 1px solid;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .signal-item.critical {
      background: var(--red-dim);
      border-color: var(--red-border);
    }

    .signal-item.warning {
      background: var(--yellow-dim);
      border-color: var(--yellow-border);
    }

    .signal-header {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .sev-badge {
      font-size: 0.68rem;
      font-weight: 700;
      font-family: var(--font-mono);
      letter-spacing: 0.08em;
      padding: 2px 8px;
      border-radius: 4px;
    }

    .sev-critical {
      background: var(--red);
      color: #fff;
    }

    .sev-warning {
      background: var(--yellow);
      color: #000;
    }

    .signal-source {
      font-size: 0.72rem;
      color: var(--text-tertiary);
    }

    .signal-technical {
      font-size: 0.78rem;
      font-family: var(--font-mono);
      color: var(--text-secondary);
      padding: 6px 10px;
      background: rgba(0,0,0,0.25);
      border-radius: var(--r-sm);
      word-break: break-word;
    }

    .signal-plain {
      font-size: 0.87rem;
      color: var(--text-primary);
      line-height: 1.6;
    }
  `],
})
export class FraudSignalsComponent {
  @Input() explanations: Explanation[] = [];
}
