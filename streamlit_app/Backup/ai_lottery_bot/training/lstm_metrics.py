#!/usr/bin/env python3
"""
LSTM Lottery-Aware Metrics
Custom metrics and loss functions for proper LSTM lottery training evaluation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from typing import List, Tuple

class LotteryAwareAccuracy(Metric):
    """Custom metric that evaluates LSTM predictions like lottery numbers"""
    
    def __init__(self, name='lottery_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_correct = self.add_weight(name='total_correct', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update state with lottery-aware evaluation"""
        # Round predictions to nearest integers
        y_pred_rounded = tf.round(tf.clip_by_value(y_pred, 1, 50))
        y_true_rounded = tf.round(tf.clip_by_value(y_true, 1, 50))
        
        # Calculate batch accuracy using set intersection approach
        batch_size = tf.shape(y_true)[0]
        num_features = tf.shape(y_true)[1]
        
        # For each sample in batch, calculate set-based accuracy
        sample_accuracies = []
        for i in range(batch_size):
            true_nums = tf.cast(y_true_rounded[i], tf.int32)
            pred_nums = tf.cast(y_pred_rounded[i], tf.int32)
            
            # Count matches (intersection size)
            matches = tf.reduce_sum(
                tf.cast(tf.equal(
                    tf.expand_dims(true_nums, 1),
                    tf.expand_dims(pred_nums, 0)
                ), tf.float32)
            )
            
            # Accuracy is matches / total numbers
            accuracy = matches / tf.cast(num_features, tf.float32)
            sample_accuracies.append(accuracy)
        
        batch_accuracy = tf.reduce_mean(sample_accuracies)
        
        self.total_correct.assign_add(batch_accuracy * tf.cast(batch_size, tf.float32))
        self.total_samples.assign_add(tf.cast(batch_size, tf.float32))
    
    def result(self):
        return tf.divide_no_nan(self.total_correct, self.total_samples)
    
    def reset_states(self):
        self.total_correct.assign(0.0)
        self.total_samples.assign(0.0)


class ExactRowAccuracy(Metric):
    """Metric that matches the 88% exact row accuracy calculation"""
    
    def __init__(self, name='exact_row_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.exact_matches = self.add_weight(name='exact_matches', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update state with exact row matching"""
        # Process predictions like the actual prediction system
        y_pred_processed = self._process_predictions(y_pred)
        y_true_processed = self._process_predictions(y_true)
        
        # Check for exact matches (all numbers in a row match)
        exact_matches = tf.reduce_all(
            tf.equal(y_true_processed, y_pred_processed), axis=1
        )
        
        self.exact_matches.assign_add(tf.reduce_sum(tf.cast(exact_matches, tf.float32)))
        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    
    def _process_predictions(self, predictions):
        """Process raw predictions like the real system"""
        # Round and clip to valid range [1, 50]
        processed = tf.round(predictions)
        processed = tf.clip_by_value(processed, 1, 50)
        
        # Sort each row to match evaluation methodology  
        processed = tf.sort(processed, axis=1)
        
        return tf.cast(processed, tf.int32)
    
    def result(self):
        return tf.divide_no_nan(self.exact_matches, self.total_samples)
    
    def reset_states(self):
        self.exact_matches.assign(0.0)
        self.total_samples.assign(0.0)


def lottery_aware_loss(y_true, y_pred):
    """Custom loss function that understands lottery number constraints"""
    # Standard MSE for continuous training signal
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    # Penalty for predictions outside valid range [1, 50]
    lower_bound_penalty = tf.reduce_mean(tf.maximum(0.0, 1.0 - y_pred))
    upper_bound_penalty = tf.reduce_mean(tf.maximum(0.0, y_pred - 50.0))
    range_penalty = lower_bound_penalty + upper_bound_penalty
    
    # Penalty for duplicate predictions within each sample
    y_pred_rounded = tf.round(tf.clip_by_value(y_pred, 1, 50))
    
    # Calculate diversity penalty (encourage unique numbers per row)
    batch_size = tf.shape(y_pred)[0]
    num_features = tf.shape(y_pred)[1]
    
    diversity_penalties = []
    for i in range(batch_size):
        row = y_pred_rounded[i]
        unique_values, _ = tf.unique(row)
        num_unique = tf.shape(unique_values)[0]
        expected_unique = tf.cast(num_features, tf.int32)
        
        # Penalty for having fewer unique values than expected
        diversity_penalty = tf.maximum(0, expected_unique - num_unique)
        diversity_penalties.append(tf.cast(diversity_penalty, tf.float32))
    
    avg_diversity_penalty = tf.reduce_mean(diversity_penalties)
    
    # Combine losses with weights
    total_loss = mse_loss + 0.1 * range_penalty + 0.05 * avg_diversity_penalty
    
    return total_loss


def lottery_mae_loss(y_true, y_pred):
    """Alternative lottery-aware loss using MAE"""
    # Convert to lottery-valid predictions
    y_pred_clipped = tf.clip_by_value(y_pred, 1, 50)
    y_true_clipped = tf.clip_by_value(y_true, 1, 50)
    
    # Use MAE as base loss
    mae_loss = tf.keras.losses.mean_absolute_error(y_true_clipped, y_pred_clipped)
    
    # Add range penalty
    range_penalty = tf.reduce_mean(
        tf.maximum(0.0, 1.0 - y_pred) + tf.maximum(0.0, y_pred - 50.0)
    )
    
    return mae_loss + 0.1 * range_penalty