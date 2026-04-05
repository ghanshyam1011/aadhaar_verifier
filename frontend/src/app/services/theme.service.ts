import { Injectable, signal } from '@angular/core';

export type ThemeMode = 'dark' | 'claude' | 'system';

@Injectable({ providedIn: 'root' })
export class ThemeService {
  readonly theme = signal<ThemeMode>('dark');

  constructor() {
    this.theme.set(this.loadTheme());
    this.applyTheme(this.theme());
  }

  setTheme(theme: ThemeMode): void {
    this.theme.set(theme);
    localStorage.setItem('aadhaarcheck-theme', theme);
    this.applyTheme(theme);
  }

  private loadTheme(): ThemeMode {
    const stored = localStorage.getItem('aadhaarcheck-theme') as ThemeMode | null;
    if (stored === 'dark' || stored === 'claude' || stored === 'system') {
      return stored;
    }
    return 'dark';
  }

  private applyTheme(theme: ThemeMode): void {
    const root = document.documentElement;
    root.dataset['theme'] = theme;

    if (theme === 'system') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.dataset['mode'] = prefersDark ? 'dark' : 'claude';
      return;
    }

    root.dataset['mode'] = theme;
  }
}
