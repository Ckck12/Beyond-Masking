"""Makes the 'trainer' directory a package and exposes the ModelTrainer."""

from .training_pipeline import ModelTrainer

__all__ = ['ModelTrainer']