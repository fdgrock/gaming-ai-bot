from sklearn.ensemble import RandomForestClassifier
from typing import Any, Dict
import numpy as np
from sklearn.metrics import log_loss
from scipy.stats import entropy


def train_rf(X, y, **kwargs) -> Any:
    clf = RandomForestClassifier(**kwargs)
    clf.fit(X, y)
    return clf


class ModelTrainer:
    def train_baseline_model(self, features: Any, labels: Any) -> Dict[str, Any]:
        """Train a baseline RandomForest classifier on numeric features.

        Args:
            features: pandas DataFrame or numpy array of features.
            labels: array-like labels (binary or multiclass).

        Returns:
            dict with keys: model, metrics
        """
        import numpy as _np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score

        # Accept DataFrame or array
        if hasattr(features, 'select_dtypes'):
            X = features.select_dtypes(include='number').fillna(0).values
        else:
            X = _np.array(features)

        y = _np.array(labels)
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty features or labels provided to train_baseline_model")

        # Simple train/test split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average='weighted')

        metrics = {"accuracy": float(acc), "f1_weighted": float(f1)}
        return {"model": clf, "metrics": metrics}

    def train_sequence_model(self, time_series_data: Any, labels: Any) -> Dict[str, Any]:
        """Train a sequence model (LSTM/TCN)."""
        # TODO: Implement training logic
        pass

    def log_metrics(self, metrics: Dict[str, float], platform: str = "mlflow") -> None:
        """Log metrics to MLflow or W&B."""
        # TODO: Implement logging logic
        pass

    def time_series_cv(self, data, labels, horizon=5):
        """Perform expanding window validation."""
        metrics = []
        for t in range(len(data) - horizon):
            train_data, val_data = data[:t], data[t:t + horizon]
            train_labels, val_labels = labels[:t], labels[t:t + horizon]
            # Train and validate model
            model = self.train_baseline_model(train_data, train_labels)
            predictions = model.predict(val_data)
            metrics.append(log_loss(val_labels, predictions))
        return np.mean(metrics)

    def scheduled_retraining(self, data, labels, rolling_metrics, brier_threshold=0.1, drift_threshold=0.05):
        """Retrain model based on schedule or triggers."""
        if len(rolling_metrics) % 6 == 0 or rolling_metrics[-1]["brier_mean"] > brier_threshold:
            print("Scheduled retraining triggered.")
            self.train_baseline_model(data, labels)

        if self.detect_data_drift(data, drift_threshold):
            print("Data drift detected. Retraining triggered.")
            self.train_baseline_model(data, labels)

    def detect_data_drift(self, data, threshold):
        """Detect data drift using KL divergence or chi-square test."""
        recent_freq = np.histogram(data[-60:], bins=range(1, 50))[0]
        baseline_freq = np.histogram(data, bins=range(1, 50))[0]
        kl_divergence = entropy(recent_freq, baseline_freq)
        return kl_divergence > threshold

    def ensemble_governance(self, champion_model, challenger_model, metrics_list):
        """Promote challenger if it outperforms the champion."""
        if challenger_model["avg_matches"] > champion_model["avg_matches"] and \
           challenger_model["brier_mean"] < champion_model["brier_mean"]:
            print("Promoting challenger to champion.")
            return challenger_model
        return champion_model

    def diversity_control(self, candidate_sets):
        """Add diversity penalty to avoid near-duplicate sets."""
        penalties = []
        for i, set_i in enumerate(candidate_sets):
            penalty = 0
            for j, set_j in enumerate(candidate_sets):
                if i != j:
                    penalty += 1 - len(set(set_i) & set(set_j)) / len(set(set_i) | set(set_j))
            penalties.append(penalty)
        return penalties
