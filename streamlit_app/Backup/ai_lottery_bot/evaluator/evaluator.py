from sklearn.metrics import accuracy_score
from typing import Any, Dict
from datetime import date
import os
import json
import csv
import statistics


def compare_predictions(preds, actuals):
    return {"accuracy": accuracy_score(actuals, preds)}


class Evaluator:
    def evaluate(self, predictions: list, actuals: list) -> Dict[str, float]:
        """Evaluate predictions against actuals and return metrics."""
        hit_rate = sum(1 for pred in predictions if any(num in actuals for num in pred)) / len(predictions)
        exact_match_rate = sum(1 for pred in predictions if set(pred) == set(actuals)) / len(predictions)
        avg_matches = sum(len(set(pred) & set(actuals)) for pred in predictions) / len(predictions)
        return {
            "hit_rate": hit_rate,
            "exact_match_rate": exact_match_rate,
            "avg_matches": avg_matches,
        }

    def record_metrics(self, game: str, metrics: Dict[str, float]) -> None:
        """Record metrics in a persistent store."""
        metrics_dir = "metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_file = os.path.join(metrics_dir, f"{game}.csv")
        file_exists = os.path.isfile(metrics_file)
        with open(metrics_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["hit_rate", "exact_match_rate", "avg_matches"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

    def evaluate_and_record(self, game: str, draw_date: date, actuals: list) -> None:
        """Load cached predictions, evaluate, and record metrics."""
        cache_file = os.path.join("predictions", game, f"{draw_date}.json")
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cached predictions not found for {draw_date}")
        with open(cache_file, "r") as f:
            predictions = json.load(f)
        metrics = self.evaluate(predictions, actuals)
        self.record_metrics(game, metrics)

    def compute_metrics(self, actual_set, predicted_sets):
        """Compute ticket-level and draw-level metrics."""
        K = len(actual_set)
        N = len(predicted_sets)

        # Ticket-level metrics
        matches_per_set = [len(set(pred) & set(actual_set)) for pred in predicted_sets]
        best_match = max(matches_per_set)
        any_hit = int(any(m >= 1 for m in matches_per_set))
        full_match = int(any(m == K for m in matches_per_set))

        # Draw-level aggregates
        avg_matches = sum(matches_per_set) / N
        hit_rate = int(any(m >= 1 for m in matches_per_set))
        match_histogram = {i: matches_per_set.count(i) for i in range(K + 1)}

        return {
            "matches_per_set": matches_per_set,
            "best_match": best_match,
            "any_hit": any_hit,
            "full_match": full_match,
            "avg_matches": avg_matches,
            "hit_rate": hit_rate,
            "match_histogram": match_histogram,
        }

    def compute_brier_score(self, actual_set, predicted_probs):
        """Compute Brier score per number and mean Brier score."""
        brier_scores = []
        for num, prob in predicted_probs.items():
            y_j = 1 if num in actual_set else 0
            brier_scores.append((prob - y_j) ** 2)
        brier_mean = sum(brier_scores) / len(brier_scores)
        return brier_mean

    def compute_rolling_stats(self, metrics_list, window=7):
        """Compute rolling averages for metrics over a given window."""
        rolling_avg_matches = sum(m["avg_matches"] for m in metrics_list[-window:]) / window
        rolling_hit_rate = sum(m["hit_rate"] for m in metrics_list[-window:]) / window
        rolling_brier_mean = sum(m["brier_mean"] for m in metrics_list[-window:]) / window
        return {
            "rolling_avg_matches": rolling_avg_matches,
            "rolling_hit_rate": rolling_hit_rate,
            "rolling_brier_mean": rolling_brier_mean,
        }

    def evaluate_and_store(self, game, draw_date, actual_set):
        """Evaluate predictions and store metrics."""
        predictions_file = os.path.join("predictions", game, f"{draw_date}.json")
        metrics_file = os.path.join("metrics", f"{game}.csv")

        if not os.path.exists(predictions_file):
            raise FileNotFoundError(f"Predictions not found for {draw_date}")

        with open(predictions_file, "r") as f:
            predictions = json.load(f)

        predicted_sets = predictions["sets"]
        per_number_probs = predictions["per_number_probs"]
        model_version = predictions["model_version"]
        num_sets = predictions["num_sets"]
        K = len(actual_set)

        # Compute metrics
        metrics = self.compute_metrics(actual_set, predicted_sets)
        brier_mean = self.compute_brier_score(actual_set, per_number_probs)
        metrics["brier_mean"] = brier_mean

        # Append metrics to CSV
        file_exists = os.path.isfile(metrics_file)
        with open(metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["draw_date", "model_version", "num_sets", "K", "best_match", "avg_matches", "any_hit", "full_match", "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "brier_mean"])
            row = [
                draw_date, model_version, num_sets, K, metrics["best_match"], metrics["avg_matches"],
                metrics["any_hit"], metrics["full_match"],
                metrics["match_histogram"].get(0, 0), metrics["match_histogram"].get(1, 0),
                metrics["match_histogram"].get(2, 0), metrics["match_histogram"].get(3, 0),
                metrics["match_histogram"].get(4, 0), metrics["match_histogram"].get(5, 0),
                metrics["match_histogram"].get(6, 0), metrics["match_histogram"].get(7, 0),
                brier_mean
            ]
            writer.writerow(row)

        return metrics

    def emit_alerts(self, metrics_list, rolling_window=7, std_dev_threshold=2, brier_degradation=0.1):
        """Emit alerts based on rolling stats and degradation thresholds."""
        rolling_stats = self.compute_rolling_stats(metrics_list, window=rolling_window)
        recent_median = statistics.mean(m["avg_matches"] for m in metrics_list[-rolling_window:])
        recent_std_dev = statistics.stdev(m["avg_matches"] for m in metrics_list[-rolling_window:])

        if rolling_stats["rolling_avg_matches"] < (recent_median - std_dev_threshold * recent_std_dev):
            print("ALERT: Rolling avg_matches dropped significantly!")

        if rolling_stats["rolling_brier_mean"] > (1 + brier_degradation) * metrics_list[-1]["brier_mean"]:
            print("ALERT: Brier mean degraded significantly!")
