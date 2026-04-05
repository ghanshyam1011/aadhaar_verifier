import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import { VerificationResponse } from '../models/verification.model';

export interface UploadPayload {
  front:  File;
  back?:  File | null;
  selfie?: File | null;
}

@Injectable({ providedIn: 'root' })
export class VerificationService {
  private http = inject(HttpClient);
  private base = environment.apiUrl;

  verify(payload: UploadPayload): Observable<VerificationResponse> {
    const fd = new FormData();
    fd.append('front', payload.front);
    if (payload.back)   fd.append('back',   payload.back);
    if (payload.selfie) fd.append('selfie', payload.selfie);

    return this.http.post<VerificationResponse>(`${this.base}/validate`, fd).pipe(
      catchError(this.handleError),
    );
  }

  getStats(): Observable<any> {
    return this.http.get(`${this.base}/stats`).pipe(catchError(this.handleError));
  }

  getHealth(): Observable<any> {
    return this.http.get(`${this.base}/health`).pipe(catchError(this.handleError));
  }

  reportUrl(reportId: string): string {
    return `${this.base}/report/${reportId}`;
  }

  private handleError(err: HttpErrorResponse): Observable<never> {
    let msg = 'An unexpected error occurred.';
    if (err.error && typeof err.error === 'object') {
      msg = err.error.plain || err.error.error || err.error.details || msg;
    } else if (err.status === 0) {
      msg = 'Cannot connect to server. Make sure the backend is running on port 5000.';
    } else if (err.status === 409) {
      msg = err.error?.plain || 'Duplicate card submission detected.';
    } else if (err.status === 429) {
      msg = err.error?.error || 'Rate limit exceeded. Please wait before retrying.';
    } else if (err.status === 400) {
      msg = err.error?.plain || err.error?.error || 'Invalid input.';
    }
    return throwError(() => new Error(msg));
  }
}
