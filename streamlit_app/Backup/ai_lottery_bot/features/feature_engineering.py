"""Feature engineering helpers for lottery draws.

Contract:
- Input `draws` is a list where each element is either:
  - a list/tuple of drawn numbers (main numbers, optionally including bonus at end), or
  - a dict containing keys: 'numbers' (list or comma-string), optional 'bonus', optional 'draw_date' (ISO str), optional 'jackpot'.

- The generator computes per-draw features and returns a list of dicts, one per input draw.
"""
from __future__ import annotations

import math
import statistics as _st
from datetime import datetime
from typing import List, Dict, Any, Optional, Sequence, Union


NumberSeq = Sequence[int]
DrawInput = Union[NumberSeq, Dict[str, Any]]


def _to_list(d: DrawInput) -> Dict[str, Any]:
    """Normalize a draw input into a dict with keys: numbers (list), bonus (optional), draw_date (optional), jackpot (optional)."""
    out = {'numbers': [], 'bonus': None, 'draw_date': None, 'jackpot': None}
    if isinstance(d, dict):
        nums = d.get('numbers')
        if isinstance(nums, str):
            nums = [int(x) for x in nums.split(',') if x.strip().isdigit()]
        out['numbers'] = list(nums) if nums is not None else []
        out['bonus'] = d.get('bonus') if d.get('bonus') is not None else None
        out['draw_date'] = d.get('draw_date')
        out['jackpot'] = d.get('jackpot')
    else:
        # assume sequence of ints
        out['numbers'] = list(d)
    return out


def _parse_jackpot(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).replace('$', '').replace(',', '').strip().lower()
        # handle million shorthand like '5.5m' or '5m' or '5 million'
        if 'm' in s or 'million' in s:
            s = s.replace('million', '').replace('m', '').strip()
            return float(s) * 1_000_000
        return float(s)
    except Exception:
        return None


def generate_features(draws: List[DrawInput], *, pool_size: int = 50, main_count: Optional[int] = None, lookback_list: Sequence[int] = (30, 60, 100), include_one_hot: bool = False, include_targets: bool = False) -> List[Dict[str, Any]]:
    """Compute a rich set of per-draw features.

    - draws should be ordered oldest -> newest. For lookback-based features the function uses prior draws.
    - pool_size controls one-hot vector length (49 or 50).
    - main_count: if provided, number of main numbers in a draw (e.g. 6 for 6/49). If None we infer: if there's a bonus field we treat last number as bonus, else assume all are main.
    """
    norm = [_to_list(d) for d in draws]

    # build quick lookup of historical occurrences for frequency/recency
    last_seen = {i: None for i in range(1, pool_size + 1)}  # map number->index of last seen draw (0-based)
    # list of feature dicts
    out_features: List[Dict[str, Any]] = []

    # maintain rolling frequency windows (as list of numbers for prior draws)
    prior_numbers: List[List[int]] = []

    for idx, rec in enumerate(norm):
        nums = rec.get('numbers', [])
        # determine main/bonus separation
        bonus = rec.get('bonus')
        if bonus is None and main_count is not None and len(nums) > main_count:
            # assume last is bonus
            main = nums[:main_count]
            bonus = nums[main_count]
        else:
            # if dict provided contained bonus key, respect it (do not strip the numbers list)
            if rec.get('bonus') is not None:
                main = [n for n in nums if n != rec.get('bonus')]
            else:
                main = nums.copy()

        main_sorted = sorted(main)

        f: Dict[str, Any] = {}

        # Numerical set features
        f['sum_all'] = sum(nums) if nums else None
        f['sum_main'] = sum(main_sorted) if main_sorted else None
        f['mean'] = float(_st.mean(main_sorted)) if main_sorted else None
        f['median'] = float(_st.median(main_sorted)) if main_sorted else None
        f['min'] = int(min(main_sorted)) if main_sorted else None
        f['max'] = int(max(main_sorted)) if main_sorted else None
        f['range'] = (f['max'] - f['min']) if (f['max'] is not None and f['min'] is not None) else None
        # std/var
        try:
            f['std_main'] = float(_st.pstdev(main_sorted)) if main_sorted else None
            f['var_main'] = float(_st.pvariance(main_sorted)) if main_sorted else None
        except Exception:
            f['std_main'] = None
            f['var_main'] = None

        # positional features
        f['first_number'] = int(main_sorted[0]) if main_sorted else None
        f['last_number'] = int(main_sorted[-1]) if main_sorted else None

        # deltas
        deltas = [j - i for i, j in zip(main_sorted[:-1], main_sorted[1:])] if len(main_sorted) >= 2 else []
        f['mean_delta'] = float(_st.mean(deltas)) if deltas else None
        f['max_delta'] = int(max(deltas)) if deltas else None
        try:
            f['std_delta'] = float(_st.pstdev(deltas)) if deltas else None
        except Exception:
            f['std_delta'] = None

        # odd/even
        num_odd = sum(1 for n in main_sorted if n % 2 != 0)
        num_even = sum(1 for n in main_sorted if n % 2 == 0)
        f['num_odd'] = int(num_odd)
        f['num_even'] = int(num_even)
        # backward-compatible aliases
        f['odd_count'] = int(num_odd)
        f['even_count'] = int(num_even)
        f['odd_even_ratio'] = float(num_odd / num_even) if num_even != 0 else None

        # range buckets (adjust upper bound for pool_size)
        upper = pool_size
        buckets = {
            'count_1_15': 0,
            'count_16_30': 0,
            'count_31_45': 0,
            'count_46_%d' % upper: 0,
        }
        for n in main_sorted:
            if 1 <= n <= 15:
                buckets['count_1_15'] += 1
            elif 16 <= n <= 30:
                buckets['count_16_30'] += 1
            elif 31 <= n <= 45:
                buckets['count_31_45'] += 1
            elif 46 <= n <= upper:
                buckets['count_46_%d' % upper] += 1
        f.update(buckets)

        # frequency-based features: counts of numbers that appeared in last N draws
        for N in lookback_list:
            key_hot = f'hot_match_count_{N}'
            key_cold = f'cold_match_count_{N}'
            recent = [n for past in prior_numbers[-N:] for n in past]
            recent_set = set(recent)
            f[key_hot] = int(sum(1 for n in main_sorted if n in sorted(recent_set)))
            f[key_cold] = int(sum(1 for n in main_sorted if n not in recent_set))

        # recency: days since last seen (requires draw_date) - aggregate avg
        draw_date = rec.get('draw_date')
        avg_days = None
        if draw_date:
            try:
                cur_dt = datetime.fromisoformat(str(draw_date)) if isinstance(draw_date, str) else draw_date
                days = []
                for n in main_sorted:
                    last = last_seen.get(n)
                    if last is not None and isinstance(last, datetime):
                        days.append((cur_dt - last).days)
                avg_days = float(_st.mean(days)) if days else None
            except Exception:
                avg_days = None
        f['avg_days_since_last_seen'] = avg_days

        # repeat patterns
        if prior_numbers:
            last_draw = prior_numbers[-1]
            f['num_repeat_last_draw'] = int(sum(1 for n in main_sorted if n in last_draw))
            f['bonus_same_as_last'] = int(bool(bonus and (str(bonus) in map(str, last_draw))))
            # repeats in last N draws for some defaults (30)
            N = 30
            lastN = [n for past in prior_numbers[-N:] for n in past]
            f['num_repeat_last_30_draws'] = int(sum(1 for n in main_sorted if n in set(lastN)))
        else:
            f['num_repeat_last_draw'] = 0
            f['bonus_same_as_last'] = 0
            f['num_repeat_last_30_draws'] = 0

        # date/time features
        if draw_date:
            try:
                ddt = datetime.fromisoformat(str(draw_date)) if isinstance(draw_date, str) else draw_date
                f['draw_dayofweek'] = int(ddt.weekday())
                f['draw_month'] = int(ddt.month)
                f['draw_year'] = int(ddt.year)
                # week index since epoch-like reference
                f['weeks_since_start'] = int((ddt - datetime(1970,1,1)).days // 7)
            except Exception:
                f['draw_dayofweek'] = None
                f['draw_month'] = None
                f['draw_year'] = None
                f['weeks_since_start'] = None
        else:
            f['draw_dayofweek'] = None
            f['draw_month'] = None
            f['draw_year'] = None
            f['weeks_since_start'] = None

        # jackpot features
        jval = _parse_jackpot(rec.get('jackpot'))
        f['jackpot_value'] = jval
        if prior_numbers:
            # find previous jackpot if available in prior records
            prev_j = None
            for p in reversed(norm[:idx]):
                pj = _parse_jackpot(p.get('jackpot'))
                if pj is not None:
                    prev_j = pj
                    break
            if jval is not None and prev_j is not None:
                f['jackpot_change'] = float(jval - prev_j)
                try:
                    f['jackpot_growth_rate'] = float((jval - prev_j) / prev_j) if prev_j != 0 else None
                except Exception:
                    f['jackpot_growth_rate'] = None
                # simple binning
                if jval < 10_000_000:
                    f['jackpot_bin'] = '<10M'
                elif jval < 20_000_000:
                    f['jackpot_bin'] = '10-20M'
                else:
                    f['jackpot_bin'] = '20M+'
            else:
                f['jackpot_change'] = None
                f['jackpot_growth_rate'] = None
                f['jackpot_bin'] = None
        else:
            f['jackpot_change'] = None
            f['jackpot_growth_rate'] = None
            f['jackpot_bin'] = None

        # metadata
        f['game_type'] = None
        f['draw_index'] = idx

        # optional one-hot vector (as list of ints)
        if include_one_hot:
            vec = [0] * pool_size
            for n in main_sorted:
                if 1 <= n <= pool_size:
                    vec[n - 1] = 1
            f['one_hot'] = vec

        # optional target (per-number binary labels)
        if include_targets:
            target = [0] * pool_size
            for n in main_sorted:
                if 1 <= n <= pool_size:
                    target[n - 1] = 1
            f['target_one_hot'] = target

        # append and update rolling state
        out_features.append(f)
        prior_numbers.append(main_sorted)

        # update last_seen using draw_date if present
        if draw_date:
            try:
                ddt = datetime.fromisoformat(str(draw_date)) if isinstance(draw_date, str) else draw_date
                for n in main_sorted:
                    last_seen[n] = ddt
            except Exception:
                # mark by index if no date parsing
                for n in main_sorted:
                    last_seen[n] = idx
        else:
            for n in main_sorted:
                last_seen[n] = idx

    return out_features


def generate_enhanced_lotto649_features(draws: List[DrawInput], *, window_size: int = 25, 
                                       include_advanced_patterns: bool = True) -> List[Dict[str, Any]]:
    """Generate enhanced feature set specifically optimized for Lotto 6/49 to match Lotto Max sophistication.
    
    This function creates 25-57 features per timestep instead of the basic 6 features,
    including comprehensive temporal, frequency, gap, pattern, and statistical features.
    """
    if not draws:
        return []
    
    norm = [_to_list(d) for d in draws]
    enhanced_features = []
    
    # Build comprehensive historical tracking
    number_history = {i: [] for i in range(1, 50)}  # Track when each number appeared
    draw_patterns = []
    recent_frequencies = {i: 0 for i in range(1, 50)}
    
    for idx, draw_data in enumerate(norm):
        numbers = draw_data.get('numbers', [])
        if not numbers:
            continue
            
        # Ensure we have exactly 6 numbers for Lotto 6/49
        main_numbers = sorted(numbers[:6])
        
        # Initialize enhanced feature dict
        enhanced_f = {}
        
        # === BASIC STATISTICAL FEATURES (7 features) ===
        enhanced_f['sum_total'] = sum(main_numbers)
        enhanced_f['mean_value'] = sum(main_numbers) / len(main_numbers) if main_numbers else 0
        enhanced_f['median_value'] = main_numbers[len(main_numbers)//2] if main_numbers else 0
        enhanced_f['std_deviation'] = float(_st.pstdev(main_numbers)) if len(main_numbers) > 1 else 0
        enhanced_f['range_span'] = max(main_numbers) - min(main_numbers) if main_numbers else 0
        enhanced_f['min_number'] = min(main_numbers) if main_numbers else 0
        enhanced_f['max_number'] = max(main_numbers) if main_numbers else 0
        
        # === PATTERN ANALYSIS FEATURES (8 features) ===
        enhanced_f['odd_count'] = sum(1 for n in main_numbers if n % 2 == 1)
        enhanced_f['even_count'] = sum(1 for n in main_numbers if n % 2 == 0)
        enhanced_f['low_count'] = sum(1 for n in main_numbers if n <= 24)  # 1-24
        enhanced_f['high_count'] = sum(1 for n in main_numbers if n > 24)  # 25-49
        
        # Consecutive number analysis
        consecutive_pairs = sum(1 for i in range(len(main_numbers)-1) 
                              if main_numbers[i+1] - main_numbers[i] == 1)
        enhanced_f['consecutive_pairs'] = consecutive_pairs
        
        # Number spacing analysis
        gaps = [main_numbers[i+1] - main_numbers[i] for i in range(len(main_numbers)-1)]
        enhanced_f['avg_gap'] = sum(gaps) / len(gaps) if gaps else 0
        enhanced_f['max_gap'] = max(gaps) if gaps else 0
        enhanced_f['gap_variance'] = float(_st.pvariance(gaps)) if len(gaps) > 1 else 0
        
        # === RANGE DISTRIBUTION FEATURES (4 features) ===
        enhanced_f['range_1_12'] = sum(1 for n in main_numbers if 1 <= n <= 12)
        enhanced_f['range_13_24'] = sum(1 for n in main_numbers if 13 <= n <= 24)
        enhanced_f['range_25_36'] = sum(1 for n in main_numbers if 25 <= n <= 36)
        enhanced_f['range_37_49'] = sum(1 for n in main_numbers if 37 <= n <= 49)
        
        # === FREQUENCY-BASED FEATURES (10 features) ===
        if idx > 0:  # Only calculate for draws with history
            # Hot numbers (appeared in last 10 draws)
            recent_10 = [n for draw in draw_patterns[-10:] for n in draw] if len(draw_patterns) >= 1 else []
            enhanced_f['hot_numbers_count'] = sum(1 for n in main_numbers if n in recent_10)
            
            # Cold numbers (haven't appeared in last 20 draws)
            recent_20 = [n for draw in draw_patterns[-20:] for n in draw] if len(draw_patterns) >= 1 else []
            enhanced_f['cold_numbers_count'] = sum(1 for n in main_numbers if n not in recent_20)
            
            # Overdue numbers (frequency analysis)
            total_appearances = {i: sum(1 for draw in draw_patterns if i in draw) for i in range(1, 50)}
            expected_freq = len(draw_patterns) * 6 / 49  # Expected appearances
            overdue_count = sum(1 for n in main_numbers 
                              if total_appearances.get(n, 0) < expected_freq * 0.8)
            enhanced_f['overdue_numbers_count'] = overdue_count
            
            # Due numbers (should appear soon based on frequency)
            due_count = sum(1 for n in main_numbers 
                          if total_appearances.get(n, 0) > expected_freq * 1.2)
            enhanced_f['due_numbers_count'] = due_count
            
            # Last appearance tracking
            last_seen_gaps = []
            for n in main_numbers:
                last_appearance = -1
                for i in range(len(draw_patterns)-1, -1, -1):
                    if n in draw_patterns[i]:
                        last_appearance = len(draw_patterns) - 1 - i
                        break
                if last_appearance >= 0:
                    last_seen_gaps.append(last_appearance)
            
            enhanced_f['avg_last_seen'] = sum(last_seen_gaps) / len(last_seen_gaps) if last_seen_gaps else 0
            enhanced_f['max_last_seen'] = max(last_seen_gaps) if last_seen_gaps else 0
            enhanced_f['min_last_seen'] = min(last_seen_gaps) if last_seen_gaps else 0
            
            # Repeat patterns
            if draw_patterns:
                enhanced_f['repeat_from_last'] = sum(1 for n in main_numbers if n in draw_patterns[-1])
                enhanced_f['repeat_from_2nd_last'] = sum(1 for n in main_numbers 
                                                        if len(draw_patterns) > 1 and n in draw_patterns[-2])
                enhanced_f['repeat_from_last_5'] = sum(1 for n in main_numbers 
                                                      if any(n in draw for draw in draw_patterns[-5:]))
            else:
                enhanced_f['repeat_from_last'] = 0
                enhanced_f['repeat_from_2nd_last'] = 0
                enhanced_f['repeat_from_last_5'] = 0
        else:
            # Initialize for first draw
            for key in ['hot_numbers_count', 'cold_numbers_count', 'overdue_numbers_count', 
                       'due_numbers_count', 'avg_last_seen', 'max_last_seen', 'min_last_seen',
                       'repeat_from_last', 'repeat_from_2nd_last', 'repeat_from_last_5']:
                enhanced_f[key] = 0
        
        # === TEMPORAL FEATURES (5 features) ===
        draw_date = draw_data.get('draw_date')
        if draw_date:
            try:
                if isinstance(draw_date, str):
                    dt = datetime.fromisoformat(draw_date)
                else:
                    dt = draw_date
                enhanced_f['day_of_week'] = dt.weekday()
                enhanced_f['month'] = dt.month
                enhanced_f['day_of_month'] = dt.day
                enhanced_f['quarter'] = (dt.month - 1) // 3 + 1
                enhanced_f['week_of_year'] = dt.isocalendar()[1]
            except:
                enhanced_f['day_of_week'] = 0
                enhanced_f['month'] = 1
                enhanced_f['day_of_month'] = 1
                enhanced_f['quarter'] = 1
                enhanced_f['week_of_year'] = 1
        else:
            enhanced_f['day_of_week'] = 0
            enhanced_f['month'] = 1
            enhanced_f['day_of_month'] = 1
            enhanced_f['quarter'] = 1
            enhanced_f['week_of_year'] = 1
        
        # === ADVANCED PATTERN FEATURES (8 features) ===
        if include_advanced_patterns:
            # Digit sum analysis
            digit_sum = sum(sum(int(d) for d in str(n)) for n in main_numbers)
            enhanced_f['digit_sum'] = digit_sum
            enhanced_f['digit_sum_mod_9'] = digit_sum % 9
            
            # Prime number analysis
            def is_prime(n):
                if n < 2: return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0: return False
                return True
            
            enhanced_f['prime_count'] = sum(1 for n in main_numbers if is_prime(n))
            enhanced_f['composite_count'] = 6 - enhanced_f['prime_count'] - (1 if 1 in main_numbers else 0)
            
            # Number ending analysis
            last_digits = [n % 10 for n in main_numbers]
            enhanced_f['unique_last_digits'] = len(set(last_digits))
            enhanced_f['most_common_last_digit'] = max(set(last_digits), key=last_digits.count) if last_digits else 0
            
            # Sum patterns
            enhanced_f['sum_is_even'] = 1 if enhanced_f['sum_total'] % 2 == 0 else 0
            enhanced_f['sum_hundreds'] = enhanced_f['sum_total'] // 100
        else:
            for key in ['digit_sum', 'digit_sum_mod_9', 'prime_count', 'composite_count',
                       'unique_last_digits', 'most_common_last_digit', 'sum_is_even', 'sum_hundreds']:
                enhanced_f[key] = 0
        
        # === JACKPOT AND EXTERNAL FEATURES (2 features) ===
        jackpot_val = _parse_jackpot(draw_data.get('jackpot'))
        enhanced_f['jackpot_millions'] = (jackpot_val / 1_000_000) if jackpot_val else 5.0
        enhanced_f['draw_index'] = idx
        
        # Update tracking data
        draw_patterns.append(main_numbers)
        for n in main_numbers:
            if 1 <= n <= 49:
                number_history[n].append(idx)
                recent_frequencies[n] += 1
        
        # Decay recent frequencies
        if idx > 20:
            for n in range(1, 50):
                recent_frequencies[n] *= 0.95
        
        enhanced_features.append(enhanced_f)
    
    return enhanced_features


def generate_enhanced_lotto_max_features(draws: List[DrawInput], *, window_size: int = 25, 
                                        include_advanced_patterns: bool = True) -> List[Dict[str, Any]]:
    """Generate enhanced feature set specifically optimized for Lotto Max to match Lotto 6/49 sophistication.
    
    This function creates 57 features per timestep instead of basic features,
    including comprehensive temporal, frequency, gap, pattern, and statistical features
    specifically tailored for Lotto Max (7 numbers from 1-50).
    """
    if not draws:
        return []
    
    norm = [_to_list(d) for d in draws]
    enhanced_features = []
    
    # Build comprehensive historical tracking for Lotto Max (1-50 numbers)
    number_history = {i: [] for i in range(1, 51)}  # Track when each number appeared
    draw_patterns = []
    recent_frequencies = {i: 0 for i in range(1, 51)}
    
    for idx, draw_data in enumerate(norm):
        numbers = draw_data.get('numbers', [])
        if not numbers:
            continue
            
        # Ensure we have exactly 7 numbers for Lotto Max
        main_numbers = sorted(numbers[:7])
        
        # Initialize enhanced feature dict
        enhanced_f = {}
        
        # === BASIC STATISTICAL FEATURES (7 features) ===
        enhanced_f['sum_total'] = sum(main_numbers)
        enhanced_f['mean_value'] = sum(main_numbers) / len(main_numbers) if main_numbers else 0
        enhanced_f['median_value'] = main_numbers[len(main_numbers)//2] if main_numbers else 0
        enhanced_f['std_deviation'] = float(_st.pstdev(main_numbers)) if len(main_numbers) > 1 else 0
        enhanced_f['range_span'] = max(main_numbers) - min(main_numbers) if main_numbers else 0
        enhanced_f['min_number'] = min(main_numbers) if main_numbers else 0
        enhanced_f['max_number'] = max(main_numbers) if main_numbers else 0
        
        # === PATTERN ANALYSIS FEATURES (8 features) ===
        enhanced_f['odd_count'] = sum(1 for n in main_numbers if n % 2 == 1)
        enhanced_f['even_count'] = sum(1 for n in main_numbers if n % 2 == 0)
        enhanced_f['low_count'] = sum(1 for n in main_numbers if n <= 25)  # 1-25
        enhanced_f['high_count'] = sum(1 for n in main_numbers if n > 25)  # 26-50
        
        # Consecutive number analysis
        consecutive_pairs = sum(1 for i in range(len(main_numbers)-1) 
                              if main_numbers[i+1] - main_numbers[i] == 1)
        enhanced_f['consecutive_pairs'] = consecutive_pairs
        
        # Number spacing analysis
        gaps = [main_numbers[i+1] - main_numbers[i] for i in range(len(main_numbers)-1)]
        enhanced_f['avg_gap'] = sum(gaps) / len(gaps) if gaps else 0
        enhanced_f['max_gap'] = max(gaps) if gaps else 0
        enhanced_f['gap_variance'] = float(_st.pvariance(gaps)) if len(gaps) > 1 else 0
        
        # === RANGE DISTRIBUTION FEATURES (5 features - Lotto Max specific) ===
        enhanced_f['range_1_10'] = sum(1 for n in main_numbers if 1 <= n <= 10)
        enhanced_f['range_11_20'] = sum(1 for n in main_numbers if 11 <= n <= 20)
        enhanced_f['range_21_30'] = sum(1 for n in main_numbers if 21 <= n <= 30)
        enhanced_f['range_31_40'] = sum(1 for n in main_numbers if 31 <= n <= 40)
        enhanced_f['range_41_50'] = sum(1 for n in main_numbers if 41 <= n <= 50)
        
        # === FREQUENCY-BASED FEATURES (12 features - Enhanced for Lotto Max) ===
        if idx > 0:  # Only calculate for draws with history
            # Hot numbers (appeared in last 10 draws)
            recent_10 = [n for draw in draw_patterns[-10:] for n in draw] if len(draw_patterns) >= 1 else []
            enhanced_f['hot_numbers_count'] = sum(1 for n in main_numbers if n in recent_10)
            
            # Cold numbers (haven't appeared in last 20 draws)
            recent_20 = [n for draw in draw_patterns[-20:] for n in draw] if len(draw_patterns) >= 1 else []
            enhanced_f['cold_numbers_count'] = sum(1 for n in main_numbers if n not in recent_20)
            
            # Overdue numbers (frequency analysis for 7/50 game)
            total_appearances = {i: sum(1 for draw in draw_patterns if i in draw) for i in range(1, 51)}
            expected_freq = len(draw_patterns) * 7 / 50  # Expected appearances for Lotto Max
            overdue_count = sum(1 for n in main_numbers 
                              if total_appearances.get(n, 0) < expected_freq * 0.8)
            enhanced_f['overdue_numbers_count'] = overdue_count
            
            # Due numbers (should appear soon based on frequency)
            due_count = sum(1 for n in main_numbers 
                          if total_appearances.get(n, 0) > expected_freq * 1.2)
            enhanced_f['due_numbers_count'] = due_count
            
            # Last appearance tracking
            last_seen_gaps = []
            for n in main_numbers:
                last_appearance = -1
                for i in range(len(draw_patterns)-1, -1, -1):
                    if n in draw_patterns[i]:
                        last_appearance = len(draw_patterns) - 1 - i
                        break
                if last_appearance >= 0:
                    last_seen_gaps.append(last_appearance)
            
            enhanced_f['avg_last_seen'] = sum(last_seen_gaps) / len(last_seen_gaps) if last_seen_gaps else 0
            enhanced_f['max_last_seen'] = max(last_seen_gaps) if last_seen_gaps else 0
            enhanced_f['min_last_seen'] = min(last_seen_gaps) if last_seen_gaps else 0
            
            # Repeat patterns
            if draw_patterns:
                enhanced_f['repeat_from_last'] = sum(1 for n in main_numbers if n in draw_patterns[-1])
                enhanced_f['repeat_from_2nd_last'] = sum(1 for n in main_numbers 
                                                        if len(draw_patterns) > 1 and n in draw_patterns[-2])
                enhanced_f['repeat_from_last_5'] = sum(1 for n in main_numbers 
                                                      if any(n in draw for draw in draw_patterns[-5:]))
                # Additional pattern tracking for Lotto Max
                enhanced_f['repeat_from_last_10'] = sum(1 for n in main_numbers 
                                                       if any(n in draw for draw in draw_patterns[-10:]))
                enhanced_f['unique_in_last_3'] = len(set(main_numbers) - 
                                                   set(n for draw in draw_patterns[-3:] for n in draw))
            else:
                enhanced_f['repeat_from_last'] = 0
                enhanced_f['repeat_from_2nd_last'] = 0
                enhanced_f['repeat_from_last_5'] = 0
                enhanced_f['repeat_from_last_10'] = 0
                enhanced_f['unique_in_last_3'] = 7
        else:
            # Initialize for first draw
            for key in ['hot_numbers_count', 'cold_numbers_count', 'overdue_numbers_count', 
                       'due_numbers_count', 'avg_last_seen', 'max_last_seen', 'min_last_seen',
                       'repeat_from_last', 'repeat_from_2nd_last', 'repeat_from_last_5',
                       'repeat_from_last_10', 'unique_in_last_3']:
                enhanced_f[key] = 0 if key != 'unique_in_last_3' else 7
        
        # === TEMPORAL FEATURES (5 features) ===
        draw_date = draw_data.get('draw_date')
        if draw_date:
            try:
                if isinstance(draw_date, str):
                    dt = datetime.fromisoformat(draw_date)
                else:
                    dt = draw_date
                enhanced_f['day_of_week'] = dt.weekday()
                enhanced_f['month'] = dt.month
                enhanced_f['day_of_month'] = dt.day
                enhanced_f['quarter'] = (dt.month - 1) // 3 + 1
                enhanced_f['week_of_year'] = dt.isocalendar()[1]
            except:
                enhanced_f['day_of_week'] = 0
                enhanced_f['month'] = 1
                enhanced_f['day_of_month'] = 1
                enhanced_f['quarter'] = 1
                enhanced_f['week_of_year'] = 1
        else:
            enhanced_f['day_of_week'] = 0
            enhanced_f['month'] = 1
            enhanced_f['day_of_month'] = 1
            enhanced_f['quarter'] = 1
            enhanced_f['week_of_year'] = 1
        
        # === ADVANCED PATTERN FEATURES (12 features - Enhanced for Lotto Max) ===
        if include_advanced_patterns:
            # Digit sum analysis
            digit_sum = sum(sum(int(d) for d in str(n)) for n in main_numbers)
            enhanced_f['digit_sum'] = digit_sum
            enhanced_f['digit_sum_mod_9'] = digit_sum % 9
            enhanced_f['digit_sum_mod_7'] = digit_sum % 7  # Additional for 7-number game
            
            # Prime number analysis
            def is_prime(n):
                if n < 2: return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0: return False
                return True
            
            enhanced_f['prime_count'] = sum(1 for n in main_numbers if is_prime(n))
            enhanced_f['composite_count'] = 7 - enhanced_f['prime_count'] - (1 if 1 in main_numbers else 0)
            
            # Number ending analysis
            last_digits = [n % 10 for n in main_numbers]
            enhanced_f['unique_last_digits'] = len(set(last_digits))
            enhanced_f['most_common_last_digit'] = max(set(last_digits), key=last_digits.count) if last_digits else 0
            
            # Sum patterns
            enhanced_f['sum_is_even'] = 1 if enhanced_f['sum_total'] % 2 == 0 else 0
            enhanced_f['sum_hundreds'] = enhanced_f['sum_total'] // 100
            
            # Lotto Max specific patterns (7 numbers)
            enhanced_f['sum_divisible_by_7'] = 1 if enhanced_f['sum_total'] % 7 == 0 else 0
            enhanced_f['numbers_divisible_by_5'] = sum(1 for n in main_numbers if n % 5 == 0)
            enhanced_f['numbers_ending_in_0'] = sum(1 for n in main_numbers if n % 10 == 0)
        else:
            for key in ['digit_sum', 'digit_sum_mod_9', 'digit_sum_mod_7', 'prime_count', 'composite_count',
                       'unique_last_digits', 'most_common_last_digit', 'sum_is_even', 'sum_hundreds',
                       'sum_divisible_by_7', 'numbers_divisible_by_5', 'numbers_ending_in_0']:
                enhanced_f[key] = 0
        
        # === JACKPOT AND EXTERNAL FEATURES (5 features - Enhanced for Lotto Max) ===
        jackpot_val = _parse_jackpot(draw_data.get('jackpot'))
        enhanced_f['jackpot_millions'] = (jackpot_val / 1_000_000) if jackpot_val else 10.0  # Lotto Max typically higher
        enhanced_f['draw_index'] = idx
        enhanced_f['jackpot_tier'] = 1 if jackpot_val and jackpot_val >= 50_000_000 else 0  # 50M+ jackpot
        enhanced_f['numbers_above_40'] = sum(1 for n in main_numbers if n > 40)
        enhanced_f['spread_factor'] = (max(main_numbers) - min(main_numbers)) / 50 if main_numbers else 0
        
        # Update tracking data
        draw_patterns.append(main_numbers)
        for n in main_numbers:
            if 1 <= n <= 50:
                number_history[n].append(idx)
                recent_frequencies[n] += 1
        
        # Decay recent frequencies
        if idx > 20:
            for n in range(1, 51):
                recent_frequencies[n] *= 0.95
        
        enhanced_features.append(enhanced_f)
    
    return enhanced_features


def build_sequences_from_features(features: List[Dict[str, Any]], window: int = 10, use_one_hot: bool = False):
    """Build sequence windows from generated features.

    - features: list of feature dicts (as returned by generate_features), oldest->newest
    - window: number of timesteps per sample
    - use_one_hot: if True, use 'one_hot' field per feature as input; otherwise use numeric fields

    Returns (X, y) as numpy arrays when possible.
    """
    try:
        import numpy as _np
    except Exception:
        return None, None

    if not features or len(features) <= window:
        return _np.empty((0,)), _np.empty((0,))

    if use_one_hot:
        seqs = [_np.array(f.get('one_hot') or []) for f in features]
        if any(s.size == 0 for s in seqs):
            return None, None
        arr = _np.stack(seqs)
        X, y = [], []
        for i in range(window, arr.shape[0]):
            X.append(arr[i-window:i])
            y.append(arr[i])
        return _np.array(X), _np.array(y)
    else:
        # use numeric columns only
        import numpy as _np
        import pandas as _pd
        df = _pd.DataFrame(features)
        num = df.select_dtypes(include=['number']).fillna(0).values
        if num.shape[0] <= window:
            return _np.empty((0,)), _np.empty((0,))
        X, y = [], []
        for i in range(window, num.shape[0]):
            X.append(num[i-window:i])
            y.append(num[i])
        return _np.array(X), _np.array(y)

