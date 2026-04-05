import { Component, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { UploadComponent } from './components/upload/upload.component';
import { ThemeMode, ThemeService } from './services/theme.service';

const THEME_OPTIONS: { mode: ThemeMode; label: string; icon: string }[] = [
  { mode: 'dark', label: 'Dark', icon: '◼' },
  { mode: 'claude', label: 'Bright', icon: '◻' },
  // { mode: 'system', label: 'System', icon: '◐' },
];

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, UploadComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="app-shell">
      <!-- Header -->
      <header class="app-header">
        <div class="header-inner">
          <div class="brand">
            <div class="brand-logo">
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <rect width="20" height="20" rx="5" fill="#1f6feb"/>
                <path d="M5 14L10 6L15 14" stroke="white" stroke-width="1.8"
                      stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M7.5 11H12.5" stroke="white" stroke-width="1.8"
                      stroke-linecap="round"/>
              </svg>
            </div>
            <div>
              <div class="brand-name">AadhaarCheck</div>
              <div class="brand-sub">AI Verification System · v5</div>
            </div>
          </div>
          <div class="header-badges">
            <span class="badge badge-blue">UIDAI</span>
            <span class="badge badge-blue">OCR + AI</span>
            <span class="header-version mono">6-Phase · 28 Checks</span>
          </div>
          <div class="theme-switcher" role="group" aria-label="Theme selector">
            <button
              *ngFor="let option of themeOptions"
              type="button"
              class="theme-pill"
              [class.active]="theme === option.mode"
              (click)="setTheme(option.mode)"
            >
              <span class="theme-pill-icon">{{ option.icon }}</span>
              <span>{{ option.label }}</span>
            </button>
          </div>
        </div>
      </header>

      <!-- Main -->
      <main class="app-main">
        <app-upload />
      </main>

      <!-- Footer -->
      <footer class="app-footer">
        <span class="mono">AadhaarCheck v5</span>
        <span>·</span>
        <span>Powered by PaddleOCR · TrOCR · DeepFace · InsightFace · Groq</span>
        <span>·</span>
        <span>Not affiliated with UIDAI</span>
      </footer>
    </div>
  `,
  styles: [`
    .app-shell {
      position: relative;
      z-index: 1;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    .app-header {
      border-bottom: 1px solid var(--border-dim);
      background: rgba(7,9,13,0.85);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 100;
    }

    :host-context(html[data-theme='claude']) .app-header,
    :host-context(html[data-theme='system'][data-mode='claude']) .app-header {
      background: rgba(247, 245, 242, 0.90);
    }

    .header-inner {
      max-width: 1100px;
      margin: 0 auto;
      padding: 14px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-logo {
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .brand-name {
      font-size: 1.05rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .brand-sub {
      font-size: 0.68rem;
      color: var(--text-tertiary);
      font-family: var(--font-mono);
      margin-top: 1px;
    }

    .header-badges {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    .theme-switcher {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px;
      border: 1px solid var(--border-dim);
      border-radius: 999px;
      background: var(--bg-surface);
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }

    .theme-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 7px 12px;
      border: none;
      border-radius: 999px;
      background: transparent;
      color: var(--text-secondary);
      font-family: var(--font-mono);
      font-size: 0.68rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      cursor: pointer;
      transition: all var(--t-fast);

      &:hover {
        color: var(--text-primary);
        background: rgba(255, 255, 255, 0.04);
      }

      &.active {
        color: var(--text-primary);
        background: var(--accent-dim);
        box-shadow: inset 0 0 0 1px rgba(31, 111, 235, 0.22);
      }
    }

    .theme-pill-icon {
      font-size: 0.75rem;
      line-height: 1;
    }

    .header-version {
      font-size: 0.68rem;
      color: var(--text-tertiary);
    }

    .app-main {
      flex: 1;
      max-width: 1100px;
      width: 100%;
      margin: 0 auto;
      padding: 40px 24px 60px;
    }

    .app-footer {
      border-top: 1px solid var(--border-dim);
      text-align: center;
      padding: 16px 24px;
      font-size: 0.72rem;
      color: var(--text-tertiary);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    :host-context(html[data-theme='claude']) .app-footer,
    :host-context(html[data-theme='system'][data-mode='claude']) .app-footer {
      background: rgba(255, 255, 255, 0.6);
    }

    @media (max-width: 600px) {
      .header-badges { display: none; }
      .theme-switcher {
        width: 100%;
        justify-content: space-between;
      }
      .theme-pill {
        flex: 1;
        justify-content: center;
        padding-inline: 8px;
      }
      .app-main { padding: 24px 16px 40px; }
    }
  `],
})
export class AppComponent {
  readonly themeOptions = THEME_OPTIONS;

  constructor(private themeService: ThemeService) {}

  get theme(): ThemeMode {
    return this.themeService.theme();
  }

  setTheme(theme: ThemeMode): void {
    this.themeService.setTheme(theme);
  }
}
