import pandas as pd
from typing import List, Dict


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing: drop NA, ensure types."""
    df = df.copy()
    df = df.dropna()
    return df


class Preprocessor:
    def clean_and_normalize(self, numbers: List[int]) -> List[int]:
        """Clean and normalize the numbers."""
        # TODO: Implement cleaning and normalization logic
        return sorted(numbers)

    def compute_features(self, numbers: List[int]) -> Dict[str, float]:
        """Compute features like odd/even ratios, sums, etc."""
        features = {
            "odd_ratio": sum(1 for n in numbers if n % 2 != 0) / len(numbers),
            "even_ratio": sum(1 for n in numbers if n % 2 == 0) / len(numbers),
            "sum": sum(numbers),
            # TODO: Add more features like rolling frequencies, jackpot tiers
        }
        return features
