#!/usr/bin/env python
"""Test game name normalization."""

from tools.prediction_engine import ProbabilityGenerator, PredictionEngine

# Test 1: Direct normalization
print("=" * 50)
print("Test 1: Game Name Normalization")
print("=" * 50)
game = "Lotto 6/49"
normalized = game.lower().replace(" ", "_").replace("/", "_")
print(f"Input: '{game}'")
print(f"Normalized: '{normalized}'")
print(f"Expected: 'lotto_6_49'")
print(f"✓ Match: {normalized == 'lotto_6_49'}\n")

# Test 2: ProbabilityGenerator
print("=" * 50)
print("Test 2: ProbabilityGenerator Initialization")
print("=" * 50)
try:
    pg = ProbabilityGenerator("Lotto 6/49")
    print(f"✓ ProbabilityGenerator('Lotto 6/49') succeeded")
    print(f"  game_lower: {pg.game_lower}")
    print(f"  game_config keys: {list(pg.game_config.keys())}")
except Exception as e:
    print(f"✗ Error: {e}")

print()

# Test 3: PredictionEngine
print("=" * 50)
print("Test 3: PredictionEngine Initialization")
print("=" * 50)
try:
    engine = PredictionEngine("Lotto 6/49")
    print(f"✓ PredictionEngine('Lotto 6/49') succeeded")
    print(f"  game_lower: {engine.game_lower}")
except Exception as e:
    print(f"✗ Error: {e}")
