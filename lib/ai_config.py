"""
AI Configuration for SURAPP

Handles configuration for AI-powered validation using Ollama.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AIConfig:
    """Configuration for AI validation."""

    # Ollama server settings
    host: str = "http://localhost:11434"
    model: str = "llama3.2-vision"

    # Validation settings
    enabled: bool = True
    max_retries: int = 3
    timeout: int = 120  # seconds

    # Confidence thresholds
    confidence_threshold: float = 0.7  # Minimum confidence to accept result

    # Prompts
    validation_prompt: str = """Analyze these two images of a Kaplan-Meier survival curve plot.

IMAGE 1: The original plot image
IMAGE 2: The extracted curves overlaid on the original

Your task is to validate if the extraction is accurate. Check:
1. Do the extracted colored lines follow the original curves precisely?
2. Are there any sections where the extraction deviates from the original?
3. Are all curves from the original captured in the extraction?
4. Do the curves start at the correct position (usually survival=1.0 at time=0)?
5. Are the curve endpoints correctly captured?

Respond in this exact format:
MATCH: [YES/NO/PARTIAL]
CONFIDENCE: [0.0-1.0]
ISSUES: [List any specific issues found, or "None" if perfect match]
SUGGESTIONS: [Parameter adjustments if needed, or "None"]
"""

    @classmethod
    def from_environment(cls) -> "AIConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            model=os.environ.get("AI_MODEL", "llama3.2-vision"),
            enabled=os.environ.get("AI_ENABLED", "true").lower() == "true",
            max_retries=int(os.environ.get("AI_MAX_RETRIES", "3")),
            timeout=int(os.environ.get("AI_TIMEOUT", "120")),
            confidence_threshold=float(os.environ.get("AI_CONFIDENCE_THRESHOLD", "0.7")),
        )


@dataclass
class ValidationResult:
    """Result from AI validation."""

    match: str  # YES, NO, or PARTIAL
    confidence: float
    issues: list[str]
    suggestions: list[str]
    raw_response: str

    @property
    def is_valid(self) -> bool:
        """Check if extraction is considered valid."""
        return self.match == "YES" or (self.match == "PARTIAL" and self.confidence >= 0.7)

    @property
    def needs_retry(self) -> bool:
        """Check if extraction should be retried with adjusted parameters."""
        return self.match == "NO" or (self.match == "PARTIAL" and self.confidence < 0.7)

    def __str__(self) -> str:
        """Human-readable representation."""
        status = "VALID" if self.is_valid else "NEEDS IMPROVEMENT"
        lines = [
            f"Validation Result: {status}",
            f"  Match: {self.match}",
            f"  Confidence: {self.confidence:.1%}",
        ]
        if self.issues and self.issues != ["None"]:
            lines.append("  Issues:")
            for issue in self.issues:
                lines.append(f"    - {issue}")
        if self.suggestions and self.suggestions != ["None"]:
            lines.append("  Suggestions:")
            for suggestion in self.suggestions:
                lines.append(f"    - {suggestion}")
        return "\n".join(lines)


@dataclass
class ExtractionParameters:
    """Adjustable parameters for curve extraction."""

    # Color detection
    saturation_min: int = 15
    value_min: int = 40
    hue_tolerance: int = 15

    # Morphological operations
    morph_kernel_size: int = 3

    # Overlap detection
    overlap_threshold: int = 3

    # Curve filtering
    min_curve_points: int = 10

    def adjust_from_suggestions(self, suggestions: list[str]) -> "ExtractionParameters":
        """Create new parameters based on AI suggestions."""
        new_params = ExtractionParameters(
            saturation_min=self.saturation_min,
            value_min=self.value_min,
            hue_tolerance=self.hue_tolerance,
            morph_kernel_size=self.morph_kernel_size,
            overlap_threshold=self.overlap_threshold,
            min_curve_points=self.min_curve_points,
        )

        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()

            # Adjust saturation threshold
            if "saturation" in suggestion_lower:
                if "lower" in suggestion_lower or "decrease" in suggestion_lower:
                    new_params.saturation_min = max(5, new_params.saturation_min - 5)
                elif "higher" in suggestion_lower or "increase" in suggestion_lower:
                    new_params.saturation_min = min(50, new_params.saturation_min + 5)

            # Adjust color tolerance
            if "color" in suggestion_lower and "tolerance" in suggestion_lower:
                if "wider" in suggestion_lower or "increase" in suggestion_lower:
                    new_params.hue_tolerance = min(30, new_params.hue_tolerance + 5)
                elif "narrower" in suggestion_lower or "decrease" in suggestion_lower:
                    new_params.hue_tolerance = max(5, new_params.hue_tolerance - 5)

            # Adjust overlap detection
            if "overlap" in suggestion_lower:
                if "stricter" in suggestion_lower or "decrease" in suggestion_lower:
                    new_params.overlap_threshold = max(1, new_params.overlap_threshold - 1)
                elif "looser" in suggestion_lower or "increase" in suggestion_lower:
                    new_params.overlap_threshold = min(10, new_params.overlap_threshold + 1)

        return new_params

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "saturation_min": self.saturation_min,
            "value_min": self.value_min,
            "hue_tolerance": self.hue_tolerance,
            "morph_kernel_size": self.morph_kernel_size,
            "overlap_threshold": self.overlap_threshold,
            "min_curve_points": self.min_curve_points,
        }
