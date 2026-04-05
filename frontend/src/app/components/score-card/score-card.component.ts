import { Component, Input, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface ScoreCardData {
  label:    string;
  score:    number;
  verdict:  string;
  passed:   boolean;
  icon:     string;
  details?: { label: string; score: number; note: string; passed: boolean }[];
}

@Component({
  selector: 'app-score-card',
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="score-card" [class.card-pass]="data.passed" [class.card-fail]="!data.passed">
      <div class="card-top">
        <div class="card-left">
          <div class="card-icon">{{ data.icon }}</div>
          <div>
            <div class="card-label">{{ data.label }}</div>
            <div class="card-verdict" [class.verdict-pass]="data.passed" [class.verdict-fail]="!data.passed">
              {{ data.verdict }}
            </div>
          </div>
        </div>
        <div class="score-ring-wrap">
          <svg width="64" height="64" viewBox="0 0 64 64">
            <circle cx="32" cy="32" r="26" fill="none"
                    stroke="rgba(255,255,255,0.06)" stroke-width="5"/>
            <circle cx="32" cy="32" r="26" fill="none"
                    [attr.stroke]="ringColor"
                    stroke-width="5"
                    stroke-linecap="round"
                    [attr.stroke-dasharray]="163"
                    [attr.stroke-dashoffset]="dashOffset"
                    stroke-dashoffset="163"
                    style="transform:rotate(-90deg);transform-origin:50% 50%;
                           transition:stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1)"
                    class="ring-fill"
            />
          </svg>
          <div class="ring-value mono" [style.color]="ringColor">{{ data.score }}</div>
        </div>
      </div>

      <!-- Progress bar -->
      <div class="progress-wrap" style="margin-top:12px">
        <div class="progress-fill"
             [class.green]="data.score >= 75"
             [class.yellow]="data.score >= 50 && data.score < 75"
             [class.red]="data.score < 50"
             [style.width.%]="data.score">
        </div>
      </div>

      <!-- Detector details -->
      <div class="details" *ngIf="data.details?.length && expanded">
        <div class="detail-row" *ngFor="let d of data.details">
          <span class="detail-dot" [class.dot-pass]="d.passed" [class.dot-fail]="!d.passed"></span>
          <span class="detail-label">{{ d.label }}</span>
          <span class="detail-score mono">{{ d.score }}</span>
          <span class="detail-note">{{ d.note | slice:0:40 }}</span>
        </div>
      </div>

      <button class="expand-btn" *ngIf="data.details?.length" (click)="expanded = !expanded">
        {{ expanded ? '▲ Hide details' : '▼ Show ' + data.details!.length + ' checks' }}
      </button>
    </div>
  `,
  styles: [`
    .score-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-dim);
      border-radius: var(--r-lg);
      padding: 18px 20px;
      transition: border-color 0.2s;
    }
    .card-pass { border-left: 3px solid var(--green); }
    .card-fail { border-left: 3px solid var(--red); }

    .card-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .card-left {
      display: flex;
      align-items: center;
      gap: 12px;
      flex: 1;
    }

    .card-icon { font-size: 1.4rem; }

    .card-label {
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--text-secondary);
    }

    .card-verdict {
      font-size: 0.9rem;
      font-weight: 600;
      margin-top: 2px;
    }
    .verdict-pass { color: var(--green); }
    .verdict-fail { color: var(--red); }

    /* Score ring */
    .score-ring-wrap {
      position: relative;
      width: 64px; height: 64px;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .score-ring-wrap svg {
      position: absolute;
      top: 0; left: 0;
    }

    .ring-value {
      position: relative;
      z-index: 1;
      font-size: 1rem;
      font-weight: 500;
    }

    /* Details */
    .details {
      margin-top: 14px;
      border-top: 1px solid var(--border-dim);
      padding-top: 12px;
      display: flex;
      flex-direction: column;
      gap: 7px;
    }

    .detail-row {
      display: grid;
      grid-template-columns: 8px 130px 36px 1fr;
      align-items: center;
      gap: 8px;
      font-size: 0.76rem;
    }

    .detail-dot {
      width: 6px; height: 6px;
      border-radius: 50%;
      flex-shrink: 0;
    }
    .dot-pass { background: var(--green); }
    .dot-fail { background: var(--red); }

    .detail-label { color: var(--text-secondary); font-weight: 500; }
    .detail-score { color: var(--text-secondary); text-align: right; }
    .detail-note  { color: var(--text-tertiary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

    .expand-btn {
      background: none;
      border: none;
      color: var(--accent-light);
      font-size: 0.72rem;
      font-family: var(--font-mono);
      cursor: pointer;
      padding: 8px 0 0;
      text-align: left;
      transition: opacity 0.15s;
      &:hover { opacity: 0.8; }
    }
  `],
})
export class ScoreCardComponent {
  @Input({ required: true }) data!: ScoreCardData;
  expanded = false;

  get ringColor(): string {
    if (this.data.score >= 75) return 'var(--green)';
    if (this.data.score >= 50) return 'var(--yellow)';
    return 'var(--red)';
  }

  get dashOffset(): number {
    const circumference = 163;
    return circumference - (circumference * this.data.score / 100);
  }
}
