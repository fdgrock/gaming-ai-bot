from abc import ABC, abstractmethod
from typing import Any, Dict, List


class GameInterface(ABC):
    """Abstract interface that defines required methods for a lottery game implementation."""

    @abstractmethod
    def ingest(self, source: str) -> Any:
        """Ingest raw game data from a source (file, api, db)."""

    @abstractmethod
    def preprocess(self, raw: Any) -> Any:
        """Clean and normalize raw data into a standard dataframe/structure."""

    @abstractmethod
    def extract_features(self, processed: Any) -> Any:
        """Turn processed data into model-ready features."""

    @abstractmethod
    def train(self, features: Any, labels: Any) -> Any:
        """Train models and return a model artifact or trainer record."""

    @abstractmethod
    def predict(self, model: Any, input_data: Any) -> List[float]:
        """Make predictions using a trained model."""

    @abstractmethod
    def evaluate(self, predictions: Any, actuals: Any) -> Dict[str, float]:
        """Evaluate predictions against actual outcomes."""
