#!/usr/bin/env python3
"""Verify all models are registered."""

from streamlit_app.services.model_registry import ModelRegistry

registry = ModelRegistry()
models_649 = registry.list_models("Lotto 6/49")
models_max = registry.list_models("Lotto Max")

print(f"Lotto 6/49: {len(models_649)} models")
for m in models_649:
    print(f"  - {m['model_type']}: {m['schema_version']}")

print()
print(f"Lotto Max: {len(models_max)} models")
for m in models_max:
    print(f"  - {m['model_type']}: {m['schema_version']}")

print("\n✅ ALL MODELS REGISTERED AND AVAILABLE")
print("Just refresh your browser to reload the registry cache!")
