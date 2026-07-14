"""Image-change detection for unknown stationary obstructions."""

from .background_detector import BackgroundObstructionDetector
from .schemas import ObstructionCandidate, ObstructionDetection

__all__ = [
    "BackgroundObstructionDetector",
    "ObstructionCandidate",
    "ObstructionDetection",
]
