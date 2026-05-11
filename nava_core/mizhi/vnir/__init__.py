"""VNIR sub-package.

Heavy imports (VNIRPipeline, VNIREngine) are available via direct module
imports — they are NOT imported here to avoid loading onnxruntime/cv2 at
startup.
"""

from .analyzer import VNIRAnalyzer, VNIRStats
from .validation import validate_plant_id

__all__ = ["VNIRAnalyzer", "VNIRStats", "validate_plant_id"]
