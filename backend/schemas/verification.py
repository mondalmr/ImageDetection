from typing import Any, Literal

from pydantic import BaseModel, Field


class CheckResult(BaseModel):
    score: float | None = Field(description="0=original, 1=morphed; null if unavailable")
    flags: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None


class VerificationChecks(BaseModel):
    metadata: CheckResult
    pixel_anomalies: CheckResult
    frequency_analysis: CheckResult
    compression_regions: CheckResult
    ml_model: CheckResult


class VerificationResponse(BaseModel):
    verdict: Literal["likely_morphed", "likely_original", "inconclusive"]
    confidence: float = Field(ge=0.0, le=1.0)
    morph_score: float = Field(ge=0.0, le=1.0)
    checks: VerificationChecks
