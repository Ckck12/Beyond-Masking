"""Makes the 'models' directory a package and exposes the main model and utils."""

from .landmark_predictor import LandmarkGuidedFramework, create_model # <- This line causes the error if the class name doesn't match
from .utils import create_lightweight_model, create_robust_model

__all__ = [
    'LandmarkGuidedFramework',
    'create_model',
    'create_lightweight_model',
    'create_robust_model'
]