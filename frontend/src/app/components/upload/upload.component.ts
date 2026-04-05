import {
  Component, ChangeDetectionStrategy, ChangeDetectorRef,
  inject, signal, computed,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { VerificationService } from '../../services/verification.service';
import {
  VerificationResponse, VerificationStep, PIPELINE_STEPS
} from '../../models/verification.model';
import { ProgressStepsComponent } from '../progress-steps/progress-steps.component';
import { ResultComponent } from '../result/result.component';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule, ProgressStepsComponent, ResultComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.scss'],
})
export class UploadComponent {
  private svc = inject(VerificationService);
  private cdr = inject(ChangeDetectorRef);

  // Files
  frontFile   = signal<File | null>(null);
  backFile    = signal<File | null>(null);
  selfieFile  = signal<File | null>(null);

  frontPreview  = signal<string>('');
  backPreview   = signal<string>('');
  selfiePreview = signal<string>('');

  // State
  currentStep  = signal<VerificationStep>('idle');
  stepIndex    = signal<number>(-1);
  errorMsg     = signal<string>('');
  result       = signal<VerificationResponse | null>(null);
  steps        = PIPELINE_STEPS;

  isDragging   = signal<Record<string, boolean>>({});

  isProcessing = computed(() =>
    this.currentStep() !== 'idle' &&
    this.currentStep() !== 'done'  &&
    this.currentStep() !== 'error'
  );

  // File handling
  onFileSelect(slot: 'front' | 'back' | 'selfie', event: Event): void {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) this.setFile(slot, file);
  }

  onDrop(slot: 'front' | 'back' | 'selfie', event: DragEvent): void {
    event.preventDefault();
    this.isDragging.set({ ...this.isDragging(), [slot]: false });
    const file = event.dataTransfer?.files[0];
    if (file) this.setFile(slot, file);
  }

  onDragOver(slot: string, event: DragEvent): void {
    event.preventDefault();
    this.isDragging.set({ ...this.isDragging(), [slot]: true });
  }

  onDragLeave(slot: string): void {
    this.isDragging.set({ ...this.isDragging(), [slot]: false });
  }

  private setFile(slot: 'front' | 'back' | 'selfie', file: File): void {
    const url = URL.createObjectURL(file);
    if (slot === 'front')  { this.frontFile.set(file);  this.frontPreview.set(url); }
    if (slot === 'back')   { this.backFile.set(file);   this.backPreview.set(url); }
    if (slot === 'selfie') { this.selfieFile.set(file); this.selfiePreview.set(url); }
  }

  removeFile(slot: 'front' | 'back' | 'selfie'): void {
    if (slot === 'front')  { this.frontFile.set(null);  this.frontPreview.set(''); }
    if (slot === 'back')   { this.backFile.set(null);   this.backPreview.set(''); }
    if (slot === 'selfie') { this.selfieFile.set(null); this.selfiePreview.set(''); }
  }

  // Submit
  submit(): void {
    if (!this.frontFile()) {
      this.errorMsg.set('Please upload the front side of the Aadhaar card.');
      return;
    }
    this.errorMsg.set('');
    this.result.set(null);
    this.currentStep.set('loading');
    this.stepIndex.set(0);

    // Animate through steps while waiting
    this.animateSteps();

    this.svc.verify({
      front:  this.frontFile()!,
      back:   this.backFile(),
      selfie: this.selfieFile(),
    }).subscribe({
      next:  (res) => {
        this.currentStep.set('done');
        this.stepIndex.set(this.steps.length);
        this.result.set(res);
        this.cdr.markForCheck();
      },
      error: (err: Error) => {
        this.currentStep.set('error');
        this.errorMsg.set(err.message);
        this.cdr.markForCheck();
      },
    });
  }

  private animateSteps(): void {
    const delays = [600, 900, 800, 2000, 1400, 1000, 1200, 600];
    let idx = 0;
    const advance = () => {
      if (this.currentStep() === 'done' || this.currentStep() === 'error') return;
      if (idx < this.steps.length - 1) {
        idx++;
        this.stepIndex.set(idx);
        this.cdr.markForCheck();
        setTimeout(advance, delays[idx] || 800);
      }
    };
    setTimeout(advance, delays[0]);
  }

  reset(): void {
    this.frontFile.set(null);   this.frontPreview.set('');
    this.backFile.set(null);    this.backPreview.set('');
    this.selfieFile.set(null);  this.selfiePreview.set('');
    this.currentStep.set('idle');
    this.stepIndex.set(-1);
    this.errorMsg.set('');
    this.result.set(null);
  }
}
