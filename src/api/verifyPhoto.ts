export type Verdict = 'likely_morphed' | 'likely_original' | 'inconclusive';

export interface CheckResult {
  score: number | null;
  flags: string[];
  details: Record<string, unknown>;
  model?: string | null;
}

export interface VerificationChecks {
  metadata: CheckResult;
  pixel_anomalies: CheckResult;
  frequency_analysis: CheckResult;
  compression_regions: CheckResult;
  ml_model: CheckResult;
}

export interface VerifyResult {
  verdict: Verdict;
  confidence: number;
  morph_score: number;
  checks: VerificationChecks;
}

export async function verifyPhoto(file: File): Promise<VerifyResult> {
  const formData = new FormData();
  formData.append('photo', file);
  const response = await fetch('/api/verify', { method: 'POST', body: formData });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    const message =
      error && typeof error.detail === 'string'
        ? error.detail
        : 'Verification failed';
    throw new Error(message);
  }
  return response.json();
}
