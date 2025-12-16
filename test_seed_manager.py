"""Test script for SeedManager"""
from streamlit_app.services.seed_manager import SeedManager

# Initialize
sm = SeedManager()

print("Testing XGBoost seeds (should be 0, 1, 2, 3, 4):")
for i in range(5):
    seed = sm.get_next_seed("xgboost")
    print(f"  Seed {i+1}: {seed}")

print("\nTesting LightGBM seeds (should be 200, 201, 202):")
for i in range(3):
    seed = sm.get_next_seed("lightgbm")
    print(f"  Seed {i+1}: {seed}")

print("\nTesting Ensemble seeds (should be 600, 601):")
for i in range(2):
    seed = sm.get_next_seed("ensemble")
    print(f"  Seed {i+1}: {seed}")

print("\nCurrent state after generation:")
print(sm.export_state())

print("\n\nResetting XGBoost...")
sm.reset_model_seeds("xgboost")

print("Next XGBoost seed after reset (should be 0):")
print(f"  Seed: {sm.get_next_seed('xgboost')}")

print("\n\nFinal state:")
print(sm.export_state())
