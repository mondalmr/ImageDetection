import { useEffect, useRef, useState } from 'react';
import { uploadPhoto } from './api/uploadPhoto';
import { verifyPhoto, type VerifyResult } from './api/verifyPhoto';
import { MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_LABEL } from './constants';
import './App.css';

const CHECK_LABELS: Record<keyof VerifyResult['checks'], string> = {
  metadata: 'Metadata',
  pixel_anomalies: 'Pixel anomalies',
  frequency_analysis: 'Frequency / wavelet',
  compression_regions: 'Compression regions',
  ml_model: 'ML model',
};

function formatVerdict(verdict: VerifyResult['verdict']): string {
  switch (verdict) {
    case 'likely_morphed':
      return 'Likely morphed';
    case 'likely_original':
      return 'Likely original';
    default:
      return 'Inconclusive';
  }
}

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [verifyResult, setVerifyResult] = useState<VerifyResult | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const resetSelection = () => {
    setSelectedFile(null);
    setVerifyResult(null);
    setPreviewUrl((current) => {
      if (current) {
        URL.revokeObjectURL(current);
      }
      return null;
    });
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    setError(null);
    setUploadSuccess(null);
    setVerifyResult(null);

    if (!file) {
      return;
    }

    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file.');
      resetSelection();
      return;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      setError(`File size must not exceed ${MAX_FILE_SIZE_LABEL}.`);
      resetSelection();
      return;
    }

    setSelectedFile(file);
    setPreviewUrl((current) => {
      if (current) {
        URL.revokeObjectURL(current);
      }
      return URL.createObjectURL(file);
    });
  };

  const handleConfirmUpload = async () => {
    if (!selectedFile || isUploading) {
      return;
    }

    setError(null);
    setUploadSuccess(null);
    setIsUploading(true);

    try {
      const result = await uploadPhoto(selectedFile);
      setUploadSuccess(result.message);
      resetSelection();
    } catch {
      setError('Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleVerify = async () => {
    if (!selectedFile || isVerifying) {
      return;
    }

    setError(null);
    setVerifyResult(null);
    setIsVerifying(true);

    try {
      const result = await verifyPhoto(selectedFile);
      setVerifyResult(result);
    } catch (verifyError) {
      setError(
        verifyError instanceof Error
          ? verifyError.message
          : 'Verification failed. Please try again.',
      );
    } finally {
      setIsVerifying(false);
    }
  };

  const isBusy = isUploading || isVerifying;

  return (
    <main className="app">
      <section className="upload-card">
        <h1>Upload Photo</h1>
        <p className="subtitle">Any image format, max {MAX_FILE_SIZE_LABEL}</p>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="file-input"
          aria-label="Choose photo"
        />

        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => fileInputRef.current?.click()}
          disabled={isBusy}
        >
          Choose photo
        </button>

        {selectedFile && (
          <p className="file-name">{selectedFile.name}</p>
        )}

        {previewUrl && (
          <img src={previewUrl} alt="Selected preview" className="preview" />
        )}

        {error && <p className="message error">{error}</p>}
        {uploadSuccess && <p className="message success">{uploadSuccess}</p>}

        <div className="action-row">
          <button
            type="button"
            className="btn btn-verify"
            onClick={handleVerify}
            disabled={!selectedFile || isBusy}
          >
            {isVerifying ? 'Verifying...' : 'Verify image'}
          </button>

          <button
            type="button"
            className="btn btn-primary"
            onClick={handleConfirmUpload}
            disabled={!selectedFile || isBusy}
          >
            {isUploading ? 'Uploading...' : 'Confirm upload'}
          </button>
        </div>

        {verifyResult && (
          <section className="verify-results" aria-live="polite">
            <div className={`verdict-badge verdict-${verifyResult.verdict}`}>
              {formatVerdict(verifyResult.verdict)}
            </div>
            <p className="verify-summary">
              Confidence: {(verifyResult.confidence * 100).toFixed(1)}% · Morph
              score: {(verifyResult.morph_score * 100).toFixed(1)}%
            </p>

            <ul className="check-list">
              {(Object.keys(CHECK_LABELS) as Array<keyof VerifyResult['checks']>).map(
                (key) => {
                  const check = verifyResult.checks[key];
                  const scorePercent =
                    check.score === null ? null : (check.score * 100).toFixed(1);

                  return (
                    <li key={key} className="check-item">
                      <div className="check-header">
                        <span className="check-name">{CHECK_LABELS[key]}</span>
                        <span className="check-score">
                          {scorePercent === null ? 'N/A' : `${scorePercent}%`}
                        </span>
                      </div>
                      {check.score !== null && (
                        <div className="score-bar" aria-hidden="true">
                          <div
                            className="score-bar-fill"
                            style={{ width: `${check.score * 100}%` }}
                          />
                        </div>
                      )}
                      {check.flags.length > 0 && (
                        <ul className="flag-list">
                          {check.flags.map((flag) => (
                            <li key={flag}>{flag.replace(/_/g, ' ')}</li>
                          ))}
                        </ul>
                      )}
                    </li>
                  );
                },
              )}
            </ul>
          </section>
        )}
      </section>
    </main>
  );
}

export default App;
