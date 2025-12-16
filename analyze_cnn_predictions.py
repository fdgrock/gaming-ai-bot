"""
Debug script to understand why CNN position accuracies are low.
Analyzes predictions vs actual values to find patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def analyze_cnn_predictions():
    """Analyze CNN prediction patterns."""
    
    print("=" * 80)
    print("CNN PREDICTION ANALYSIS")
    print("=" * 80)
    
    # Load test data from Lotto Max
    data_dir = Path("data/lotto_max")
    csv_files = sorted(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found")
        return
    
    # Read all draws
    all_draws = []
    for f in csv_files:
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            numbers_str = str(row.get("numbers", ""))
            numbers = [int(n.strip()) for n in numbers_str.split(",") if n.strip().isdigit()]
            if len(numbers) == 7:
                all_draws.append(numbers)
    
    draws_array = np.array(all_draws)
    print(f"\n📊 Total draws: {len(draws_array)}")
    
    # Analyze number distribution by position
    print(f"\n" + "=" * 80)
    print("NUMBER DISTRIBUTION BY POSITION")
    print("=" * 80)
    
    for pos in range(7):
        numbers_at_pos = draws_array[:, pos]
        unique, counts = np.unique(numbers_at_pos, return_counts=True)
        
        print(f"\nPosition {pos+1}:")
        print(f"  Range: {numbers_at_pos.min()} - {numbers_at_pos.max()}")
        print(f"  Mean: {numbers_at_pos.mean():.1f}")
        print(f"  Std: {numbers_at_pos.std():.1f}")
        print(f"  Unique values: {len(unique)}")
        
        # Show top 5 most common
        top_indices = np.argsort(counts)[-5:][::-1]
        print(f"  Top 5 most common:")
        for idx in top_indices:
            num = unique[idx]
            count = counts[idx]
            pct = (count / len(numbers_at_pos)) * 100
            print(f"    {num}: {count} times ({pct:.1f}%)")
    
    # Check if positions are independent
    print(f"\n" + "=" * 80)
    print("POSITION INDEPENDENCE ANALYSIS")
    print("=" * 80)
    
    # Calculate correlation between positions
    correlation_matrix = np.corrcoef(draws_array.T)
    
    print(f"\nCorrelation between positions:")
    print(f"  Max correlation: {np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]):.3f}")
    print(f"  Min correlation: {np.min(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]):.3f}")
    print(f"  Avg correlation: {np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]):.3f}")
    
    # Calculate class distribution (0-49 for 1-50 numbers)
    print(f"\n" + "=" * 80)
    print("CLASS DISTRIBUTION (0-49 for numbers 1-50)")
    print("=" * 80)
    
    # Convert to 0-based classes
    classes = draws_array - 1
    all_classes = classes.flatten()
    unique_classes, class_counts = np.unique(all_classes, return_counts=True)
    
    print(f"\nTotal classes: {len(unique_classes)}/50")
    print(f"Missing classes: {50 - len(unique_classes)}")
    
    if len(unique_classes) < 50:
        all_possible = set(range(50))
        present = set(unique_classes)
        missing = all_possible - present
        print(f"Missing class numbers: {sorted(list(missing))}")
    
    # Show class imbalance
    max_count = class_counts.max()
    min_count = class_counts.min()
    print(f"\nClass imbalance:")
    print(f"  Most frequent: {max_count} occurrences")
    print(f"  Least frequent: {min_count} occurrences")
    print(f"  Ratio: {max_count/min_count:.2f}x")
    
    # Baseline accuracy calculation
    print(f"\n" + "=" * 80)
    print("BASELINE ACCURACY CALCULATIONS")
    print("=" * 80)
    
    # Use last 20% as test
    test_size = int(len(draws_array) * 0.2)
    train_draws = draws_array[:-test_size]
    test_draws = draws_array[-test_size:]
    
    print(f"\nTrain: {len(train_draws)} draws")
    print(f"Test: {len(test_draws)} draws")
    
    # Strategy 1: Predict most common in each position
    print(f"\n1. MOST COMMON PER POSITION:")
    position_accuracies_common = []
    for pos in range(7):
        train_numbers = train_draws[:, pos]
        unique, counts = np.unique(train_numbers, return_counts=True)
        most_common = unique[np.argmax(counts)]
        
        test_numbers = test_draws[:, pos]
        accuracy = np.mean(test_numbers == most_common)
        position_accuracies_common.append(accuracy)
        print(f"  Pos {pos+1}: Predict {most_common} → {accuracy:.1%} accuracy")
    
    avg_common = np.mean(position_accuracies_common)
    print(f"  Average: {avg_common:.1%}")
    
    # Strategy 2: Random baseline
    print(f"\n2. RANDOM BASELINE (1/50 each position):")
    print(f"  Expected: 2.0% per position")
    
    # Strategy 3: Frequency-weighted random
    print(f"\n3. FREQUENCY-WEIGHTED PREDICTION:")
    position_accuracies_freq = []
    for pos in range(7):
        train_numbers = train_draws[:, pos]
        unique, counts = np.unique(train_numbers, return_counts=True)
        
        # For each test sample, predict based on frequency
        test_numbers = test_draws[:, pos]
        correct = 0
        for test_num in test_numbers:
            # Check if test number is in top 20% most frequent
            threshold_count = np.percentile(counts, 80)
            top_numbers = unique[counts >= threshold_count]
            if test_num in top_numbers:
                correct += 1
        
        accuracy = correct / len(test_numbers)
        position_accuracies_freq.append(accuracy)
        print(f"  Pos {pos+1}: {accuracy:.1%} (top 20% frequent)")
    
    avg_freq = np.mean(position_accuracies_freq)
    print(f"  Average: {avg_freq:.1%}")
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nYour CNN achieved: 6.93% average position accuracy")
    print(f"Baselines:")
    print(f"  - Random: 2.0%")
    print(f"  - Most common: {avg_common:.1%}")
    print(f"  - Frequency-weighted: {avg_freq:.1%}")
    print(f"\nCNN is {6.93/2.0:.1f}x better than random")
    print(f"CNN is {6.93/avg_common*100:.1f}% of most-common baseline")
    
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print(f"""
1. CNN is learning SOME patterns (3.5x better than random)
2. But it's below the simple 'most common' baseline ({avg_common:.1%})
3. This suggests:
   - Need more diverse features (not just embeddings)
   - Need to capture number frequency patterns
   - Need position-specific information
   
Suggested fixes:
   - Add number frequency features (how often each number appears)
   - Add position-specific statistics (mean/std per position)
   - Add temporal features (recent vs old draws)
   - Use ensemble with tree models (XGBoost knows frequencies)
""")

if __name__ == "__main__":
    analyze_cnn_predictions()
