import {
  Component, Input, ChangeDetectionStrategy
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { PipelineStep } from '../../models/verification.model';

@Component({
  selector: 'app-progress-steps',
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="progress-section">
      <div class="progress-header">
        <div class="spinner" *ngIf="!isError; else errIcon"></div>
        <ng-template #errIcon><span class="err-icon">⚠</span></ng-template>
        <div>
          <div class="progress-title">{{ isError ? 'Verification failed' : 'Processing your card…' }}</div>
          <div class="progress-sub mono">{{ activeStep?.label ?? 'Initialising' }}</div>
        </div>
        <div class="progress-pct mono">{{ pct }}%</div>
      </div>

      <div class="overall-bar">
        <div class="overall-fill" [style.width.%]="pct"></div>
      </div>

      <div class="steps-list">
        <div
          *ngFor="let step of steps; let i = index"
          class="step-item"
          [class.active]="i === activeIndex && !isError"
          [class.done]="i < activeIndex && !isError"
          [class.error]="i === activeIndex && isError"
        >
          <span class="step-icon">
            <span *ngIf="i < activeIndex && !isError">✓</span>
            <span *ngIf="i === activeIndex && isError">✗</span>
            <span *ngIf="i === activeIndex && !isError" class="step-spinner"></span>
            <span *ngIf="i > activeIndex || (isError && i !== activeIndex)">{{ step.icon }}</span>
          </span>
          <span class="step-name">{{ step.label }}</span>
          <span class="step-time mono" *ngIf="i < activeIndex && !isError">done</span>
          <span class="step-time mono active-label" *ngIf="i === activeIndex && !isError">running</span>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .progress-section {
      margin-top: 36px;
      animation: fadeUp 0.3s ease forwards;
    }

    .progress-header {
      display: flex;
      align-items: center;
      gap: 14px;
      margin-bottom: 16px;
    }

    .progress-title {
      font-weight: 600;
      font-size: 1.05rem;
    }

    .progress-sub {
      font-size: 0.72rem;
      color: var(--text-tertiary);
      margin-top: 2px;
    }

    .progress-pct {
      margin-left: auto;
      font-size: 1.4rem;
      font-weight: 500;
      color: var(--accent-light);
    }

    .err-icon {
      font-size: 1.4rem;
      color: var(--yellow);
    }

    /* Overall progress bar */
    .overall-bar {
      height: 3px;
      background: var(--bg-elevated);
      border-radius: 2px;
      margin-bottom: 20px;
      overflow: hidden;
    }

    .overall-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent) 0%, var(--accent-light) 100%);
      border-radius: 2px;
      transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Steps */
    .steps-list {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }

    .step-item {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 9px 14px;
      border-radius: var(--r-md);
      background: var(--bg-surface);
      border: 1px solid var(--border-dim);
      font-size: 0.83rem;
      color: var(--text-tertiary);
      transition: all 0.2s ease;

      &.active {
        color: var(--accent-light);
        border-color: var(--accent);
        background: var(--accent-dim);
      }

      &.done {
        color: var(--green);
        border-color: var(--green-border);
        background: var(--green-dim);
      }

      &.error {
        color: var(--red);
        border-color: var(--red-border);
        background: var(--red-dim);
      }
    }

    .step-icon {
      min-width: 20px;
      text-align: center;
      font-size: 0.9rem;
    }

    .step-name { flex: 1; font-weight: 500; }

    .step-time {
      font-size: 0.65rem;
      color: var(--text-tertiary);

      &.active-label { color: var(--accent-light); }
    }

    .step-spinner {
      display: inline-block;
      width: 12px; height: 12px;
      border: 1.5px solid var(--accent-dim);
      border-top-color: var(--accent-light);
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
    }

    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(12px); }
      to   { opacity: 1; transform: translateY(0); }
    }
  `],
})
export class ProgressStepsComponent {
  @Input() steps: PipelineStep[] = [];
  @Input() activeIndex = 0;
  @Input() isError = false;

  get activeStep() {
    return this.steps[this.activeIndex] ?? null;
  }

  get pct(): number {
    if (this.steps.length === 0) return 0;
    return Math.round((this.activeIndex / this.steps.length) * 100);
  }
}
