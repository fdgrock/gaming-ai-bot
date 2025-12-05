# UNIFIED FEATURE SCHEMA SYSTEM - IMPLEMENTATION PLAN

## Executive Summary

This document outlines a **Unified Feature-Training-Prediction Synchronization System** that ensures:
1. **Feature Generation** creates reproducible, versioned feature schemas
2. **Model Training** stores exactly which features/parameters were used
3. **Prediction Generation** loads and applies the EXACT SAME feature pipeline

**Key Innovation**: Models no longer train on features and then forget what features were used. Each trained model remembers its feature schema.

---

## Current State: The Problem

### Issue 1: Feature Pipeline Fragmentation
```
Advanced Feature Generation (advanced_feature_generator.py)
├─ LSTM: 70+ features, RobustScaler, 25-window sequences
├─ Transformer: 50+ features, L2 normalization, multi-scale embeddings  
├─ CNN: 64-dim embeddings, multi-scale aggregation, L2 norm
├─ XGBoost: 77-115 features, various scalers, CSV format
├─ CatBoost: ~80 features, custom engineering
└─ LightGBM: ~80 features, optimized for speed

Data Training (data_training.py)
├─ Loads features from above ✓
├─ But doesn't record WHICH features used
├─ Trains model with unclear pipeline
└─ Saves model WITHOUT feature metadata ❌

Prediction Generation (predictions.py)
├─ Loads trained model
├─ Tries to guess what features it needs
├─ Often uses WRONG features or dimensions ❌
├─ Results: predictions fail or have low accuracy
└─ No way to verify match ❌
```

### Issue 2: Feature Parameter Drift
- LSTM window_size changed from 25 → 30? Models trained with old data fail
- Scaler type changed (StandardScaler → RobustScaler)? Predictions wrong
- Feature columns reordered? Model predictions meaningless
- No version control on features

### Issue 3: Model Type Specific Issues

**Tree Models (XGBoost, CatBoost, LightGBM)**
- Expect: Flat 2D arrays, specific number of features
- Training uses: CSV features directly
- Prediction uses: Random Gaussian noise ❌ (WRONG!)
- Issue: Predictions not from feature distribution

**Neural Networks (LSTM, CNN, Transformer)**
- LSTM expects: 3D sequences (N, window_size, features)
- CNN expects: 2D arrays reshaped to 4D (N, 1, window_size, features)
- Transformer expects: 2D embeddings (N, embedding_dim)
- Training uses: NPZ files with specific structure
- Prediction uses: Sometimes correct, sometimes wrong dimensions
- Issue: Shape mismatches during prediction

---

## Proposed Solution: Unified Feature Schema System

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   FEATURE GENERATION                        │
│  advanced_feature_generator.py + NEW FeatureSchema         │
│                                                              │
│  For each model type:                                       │
│  1. Generate features WITH FeatureSchema metadata           │
│  2. Save schema to model registry                           │
│  3. Save schema to feature files (JSON + binary)            │
└──────────────┬──────────────────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                           │
│  data_training.py + NEW ModelRegistry                       │
│                                                              │
│  1. Load features WITH embedded schema                      │
│  2. Record schema used during training                      │
│  3. Save model + schema to ModelRegistry                    │
│  4. Create model_manifest.json with full metadata           │
└──────────────┬──────────────────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────────────────────┐
│               SYNCHRONIZED PREDICTION                       │
│  predictions.py + NEW SynchronizedPredictor                │
│                                                              │
│  1. Load model from registry                                │
│  2. Extract embedded FeatureSchema                          │
│  3. Generate IDENTICAL features using saved schema          │
│  4. Validate compatibility BEFORE prediction                │
│  5. Make predictions with verified matching features        │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Specifications

### 1. FeatureSchema Class (NEW)

**Location**: `streamlit_app/services/feature_schema.py` (NEW FILE)

**Purpose**: Captures EVERYTHING needed to reproduce feature generation

```python
@dataclass
class FeatureSchema:
    """Serializable schema for feature reproducibility"""
    
    # Identity
    model_type: str  # "xgboost", "lstm", "cnn", "transformer", etc.
    game: str  # "Lotto 6/49", "Lotto Max"
    schema_version: str  # "1.0" for tracking changes
    created_at: str  # ISO timestamp
    
    # Feature Parameters
    feature_names: List[str]  # Exact column/feature names
    feature_count: int  # Total features
    feature_types: Dict[str, str]  # {feature_name: "temporal"|"statistical"|...}
    
    # Transformation Parameters
    normalization_method: str  # "StandardScaler", "MinMaxScaler", "RobustScaler", "L2", "None"
    normalization_params: Dict[str, Any]  # {scale: 0-1, mean: 0, std: 1}
    
    # Sequence/Window Parameters
    window_size: Optional[int]  # For LSTM/CNN (e.g., 25)
    lookback_periods: List[int]  # [5, 10, 30, 60, 100]
    stride: Optional[int]  # For sliding window
    
    # Embedding Parameters  
    embedding_dim: Optional[int]  # For Transformer/CNN (64, 128, etc.)
    embedding_method: Optional[str]  # "pca", "mean_pooling", "custom"
    
    # Statistical Transformations
    transformations: List[Dict[str, Any]]  # [
    #   {"name": "log", "features": ["entropy"]},
    #   {"name": "zscore", "features": ["jackpot"]},
    # ]
    
    # Data Quality
    data_shape: Tuple[int, ...]  # (samples, features) or (samples, window, features)
    data_date_range: Dict[str, str]  # {min: "2020-01-01", max: "2025-12-04"}
    missing_values_strategy: str  # "fillna_0", "interpolate", "drop"
    
    # Additional Metadata
    raw_data_version: str  # Which raw data CSVs were used
    feature_categories: List[str]  # ["temporal", "frequency", "parity", ...]
    notes: str  # Custom notes about this schema
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict"""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSchema":
        """Deserialize from dict"""
    
    def save_to_file(self, path: Path) -> None:
        """Save schema as JSON"""
    
    @classmethod
    def load_from_file(cls, path: Path) -> "FeatureSchema":
        """Load schema from JSON"""
    
    def validate_compatibility(self, other: "FeatureSchema") -> Tuple[bool, str]:
        """Check if this schema is compatible with another
        Returns: (is_compatible, reason_if_not)"""
```

### 2. ModelRegistry System (NEW)

**Location**: `streamlit_app/services/model_registry.py` (NEW FILE)

**Purpose**: Track models and their feature schemas

```python
class ModelRegistry:
    """Centralized registry for trained models and their feature schemas"""
    
    def __init__(self):
        """Initialize registry"""
        self.registry_file = Path(get_models_dir()) / "model_manifest.json"
        self.models = self._load_registry()
    
    def register_model(
        self,
        model_path: Path,
        model_type: str,
        game: str,
        feature_schema: FeatureSchema,
        metadata: Dict[str, Any]
    ) -> None:
        """Register a trained model with its feature schema"""
        entry = {
            "model_path": str(model_path),
            "model_type": model_type,
            "game": game,
            "feature_schema": feature_schema.to_dict(),
            "trained_at": datetime.now().isoformat(),
            "accuracy": metadata.get("accuracy"),
            "data_samples": metadata.get("data_samples"),
            "training_duration": metadata.get("training_duration"),
            "notes": metadata.get("notes", "")
        }
        self.models[f"{game}_{model_type}"] = entry
        self._save_registry()
    
    def get_model_schema(self, game: str, model_type: str) -> Optional[FeatureSchema]:
        """Retrieve feature schema for a model"""
        key = f"{game}_{model_type}"
        if key in self.models:
            return FeatureSchema.from_dict(self.models[key]["feature_schema"])
        return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
    
    def validate_model_compatibility(self, game: str, model_type: str) -> Dict[str, Any]:
        """Check if model's schema is compatible with current feature generation"""
```

### 3. SynchronizedPredictor Class (NEW)

**Location**: `streamlit_app/services/synchronized_predictor.py` (NEW FILE)

**Purpose**: Load model + schema and generate predictions with verified synchronization

```python
class SynchronizedPredictor:
    """Generates predictions using schema-synchronized features"""
    
    def __init__(self, game: str, model_type: str, registry: ModelRegistry):
        self.game = game
        self.model_type = model_type
        self.registry = registry
        self.model = None
        self.schema = None
        self.feature_generator = None
        
    def load_model_and_schema(self) -> Tuple[bool, str]:
        """Load model and verify schema"""
        # Load feature schema
        self.schema = self.registry.get_model_schema(self.game, self.model_type)
        if not self.schema:
            return False, f"No schema found for {self.game} {self.model_type}"
        
        # Load model
        model_path = self._get_model_path()
        if not model_path.exists():
            return False, f"Model not found at {model_path}"
        
        self.model = self._load_model_by_type(model_path)
        return True, "Model and schema loaded successfully"
    
    def generate_features_from_schema(self, raw_data: pd.DataFrame) -> Tuple[np.ndarray, bool, str]:
        """Generate features using EXACT schema from training"""
        if not self.schema:
            return None, False, "No schema loaded"
        
        # Initialize feature generator with schema parameters
        if self.model_type in ["xgboost", "catboost", "lightgbm"]:
            features = self._generate_tabular_features(raw_data)
        elif self.model_type == "lstm":
            features = self._generate_lstm_features(raw_data)
        elif self.model_type == "cnn":
            features = self._generate_cnn_features(raw_data)
        elif self.model_type == "transformer":
            features = self._generate_transformer_features(raw_data)
        
        return features, True, "Features generated successfully"
    
    def validate_feature_compatibility(self, features: np.ndarray) -> Tuple[bool, List[str]]:
        """Validate that generated features match schema"""
        warnings = []
        
        # Check shape
        if len(features.shape) != len(self.schema.data_shape):
            warnings.append(f"Shape mismatch: expected {self.schema.data_shape}, got {features.shape}")
        
        # Check feature count
        if self.model_type in ["xgboost", "catboost", "lightgbm"]:
            if features.shape[1] != self.schema.feature_count:
                warnings.append(f"Feature count: expected {self.schema.feature_count}, got {features.shape[1]}")
        
        # Check normalization
        feature_mean = np.mean(features)
        feature_std = np.std(features)
        if self.schema.normalization_method == "StandardScaler":
            if abs(feature_mean) > 0.1 or abs(feature_std - 1.0) > 0.1:
                warnings.append(f"Normalization mismatch: mean={feature_mean:.3f}, std={feature_std:.3f}")
        
        return len(warnings) == 0, warnings
    
    def predict(self, features: np.ndarray, num_predictions: int = 1) -> Dict[str, Any]:
        """Generate predictions with synchronized features"""
        if not self.model:
            return {"error": "Model not loaded"}
        
        # Validate compatibility
        compatible, warnings = self.validate_feature_compatibility(features)
        if warnings:
            app_log(f"Feature compatibility warnings: {warnings}", "warning")
        
        # Generate predictions based on model type
        if self.model_type in ["xgboost", "catboost", "lightgbm"]:
            predictions = self._predict_tree_model(features)
        elif self.model_type == "lstm":
            predictions = self._predict_lstm(features)
        elif self.model_type == "cnn":
            predictions = self._predict_cnn(features)
        elif self.model_type == "transformer":
            predictions = self._predict_transformer(features)
        
        return {
            "predictions": predictions,
            "schema_version": self.schema.schema_version,
            "compatibility_warnings": warnings,
            "feature_source": f"schema_synchronized_{self.model_type}"
        }
```

### 4. Enhanced Feature Generator

**File**: Modify `streamlit_app/services/advanced_feature_generator.py`

**Changes**:
- Add `FeatureSchema` creation for each feature generation method
- Save schema alongside features
- Track all transformations applied

```python
class AdvancedFeatureGenerator:
    # ... existing code ...
    
    def generate_xgboost_features(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, FeatureSchema]:
        """Generate XGBoost features WITH schema"""
        # ... feature generation code ...
        
        # CREATE SCHEMA
        schema = FeatureSchema(
            model_type="xgboost",
            game=self.game,
            schema_version="1.0",
            created_at=datetime.now().isoformat(),
            feature_names=list(features_df.columns),
            feature_count=len(features_df.columns),
            feature_types={col: self._infer_feature_type(col) for col in features_df.columns},
            normalization_method="None",  # XGBoost uses raw values
            feature_categories=["temporal", "distribution", "parity", "spacing", 
                              "statistical", "frequency", "bonus", "jackpot"],
            data_shape=features_df.shape,
            data_date_range={
                "min": str(features_df["draw_date"].min()),
                "max": str(features_df["draw_date"].max())
            }
        )
        
        return features_df, schema
    
    def generate_lstm_sequences(self, raw_data: pd.DataFrame) -> Tuple[np.ndarray, FeatureSchema]:
        """Generate LSTM sequences WITH schema"""
        # ... sequence generation code ...
        
        schema = FeatureSchema(
            model_type="lstm",
            game=self.game,
            schema_version="1.0",
            created_at=datetime.now().isoformat(),
            feature_count=len(feature_names),
            feature_names=feature_names,
            normalization_method="RobustScaler",
            window_size=window_size,
            lookback_periods=[5, 10, 30, 60, 100],
            transformations=[
                {"name": "rolling_mean", "features": list(frequency_features)},
                {"name": "zscore", "features": ["bonus"]}
            ],
            data_shape=sequences_array.shape,
            embedding_dim=None
        )
        
        return sequences_array, schema
    
    # Similar for CNN, Transformer, CatBoost, LightGBM
```

### 5. Enhanced Model Training

**File**: Modify `streamlit_app/pages/data_training.py` and `streamlit_app/services/advanced_model_training.py`

**Changes**:
- Accept FeatureSchema from feature generation
- Store schema during model training
- Register model in ModelRegistry

```python
def _train_advanced_model(game: str, model_type: str, ...):
    # ... existing training code ...
    
    # 1. Generate features WITH schema
    feature_gen = AdvancedFeatureGenerator(game)
    if model_type == "xgboost":
        features, schema = feature_gen.generate_xgboost_features(raw_data)
    elif model_type == "lstm":
        features, schema = feature_gen.generate_lstm_sequences(raw_data)
    # ... etc
    
    # 2. Train model
    trained_model, metadata = train_model_with_features(features, model_type)
    
    # 3. Save model + schema
    model_path = save_model(trained_model, model_type, game)
    registry.register_model(
        model_path=model_path,
        model_type=model_type,
        game=game,
        feature_schema=schema,
        metadata={
            "accuracy": metadata["accuracy"],
            "training_duration": metadata["duration"],
            "data_samples": len(features)
        }
    )
    
    st.success(f"Model trained with {schema.feature_count} features, schema registered")
```

### 6. Enhanced Prediction Generation

**File**: Modify `streamlit_app/pages/predictions.py`

**Changes**:
- Use SynchronizedPredictor instead of manual feature loading
- Show schema information in UI
- Validate synchronization before predicting

```python
def _generate_single_model_predictions(game, model_type, ...):
    # Load model + schema
    registry = ModelRegistry()
    predictor = SynchronizedPredictor(game, model_type, registry)
    success, msg = predictor.load_model_and_schema()
    
    if not success:
        st.error(f"Failed to load model: {msg}")
        return
    
    # Show schema info in UI
    with st.expander("📋 Feature Schema Information"):
        st.json({
            "model_type": predictor.schema.model_type,
            "feature_count": predictor.schema.feature_count,
            "normalization": predictor.schema.normalization_method,
            "window_size": predictor.schema.window_size,
            "schema_version": predictor.schema.schema_version
        })
    
    # Generate features using schema
    features, success, msg = predictor.generate_features_from_schema(raw_data)
    if not success:
        st.error(msg)
        return
    
    # Validate compatibility
    compatible, warnings = predictor.validate_feature_compatibility(features)
    if warnings:
        st.warning("⚠️ " + "\n".join(warnings))
    
    # Generate predictions
    result = predictor.predict(features, num_predictions=count)
    
    # ... display results ...
```

---

## UI Integration Strategy

### Data Training Tab Updates

1. **Feature Generation Section**
   - Show generated schema summary
   - Display feature statistics
   - Confirm schema saved

2. **Model Training Section**
   - Show "Schema Recording" progress
   - Display which schema is being used
   - Confirm model registered in ModelRegistry

### Predictions Tab Updates

1. **Before Generation**
   - Load schema for selected model
   - Display schema compatibility status
   - Warn if schema version mismatch

2. **During Generation**
   - Show "Using schema-synchronized features"
   - Display compatibility validation results

3. **After Generation**
   - Show feature source: "schema_synchronized_xgboost"
   - Display schema version used
   - Show any compatibility warnings

---

## Implementation Steps (Sequenced)

### Phase 1: Create Core Infrastructure
1. Create `feature_schema.py` with FeatureSchema class
2. Create `model_registry.py` with ModelRegistry class
3. Create `synchronized_predictor.py` with SynchronizedPredictor class
4. Add imports and exception handling

### Phase 2: Integrate into Feature Generation
1. Modify `advanced_feature_generator.py` to create schemas
2. Save schemas alongside feature files
3. Update feature generation functions to return (features, schema) tuples
4. Test feature generation with schema creation

### Phase 3: Integrate into Model Training
1. Modify `data_training.py` to accept and store schemas
2. Modify `advanced_model_training.py` to register models
3. Create model registry file (model_manifest.json)
4. Test model registration

### Phase 4: Integrate into Predictions
1. Modify `predictions.py` to use SynchronizedPredictor
2. Add UI elements showing schema information
3. Add validation before predictions
4. Test end-to-end synchronization

### Phase 5: Retraining and Testing
1. Retrain all models with schema synchronization
2. Test predictions with new registry
3. Verify feature compatibility warnings work
4. Document any issues found

---

## Benefits

### Immediate
- ✅ Clear visibility into feature-model sync status
- ✅ Automatic compatibility validation
- ✅ Warning system for mismatches
- ✅ Easier debugging

### Medium-term
- ✅ Can safely change feature engineering (features versioned)
- ✅ Can safely change models (old models still have their schemas)
- ✅ Clear migration path when updating
- ✅ Reproducibility: old predictions can be regenerated

### Long-term
- ✅ Enables A/B testing of feature schemas
- ✅ Enables gradual model updates
- ✅ Foundation for advanced ensemble methods
- ✅ Better model governance and tracking

---

## Risk Mitigation

### Backward Compatibility
- Old models will still work with new code
- Registry includes version information
- Graceful fallback if schema not found

### Testing Strategy
- Unit tests for FeatureSchema serialization
- Integration tests for schema-model pairing
- End-to-end tests for full prediction pipeline
- Compare predictions before/after on same features

### Rollback Plan
- Keep feature generation code unchanged
- Add new schema system alongside old system
- Gradual migration model by model
- Old system available as fallback

---

## Next Steps for User Review

### Questions to Answer:
1. **Tree Models**: Should we change prediction generation to use actual features instead of Gaussian noise?
2. **Versioning**: How should we handle major schema changes? (e.g., new features added)
3. **Backward Compat**: Should we auto-convert old models or require retraining?
4. **UI Priority**: Which UI improvements are most important for visibility?

### Optional Enhancements:
1. A/B testing system comparing different schemas
2. Feature importance tracking per schema
3. Automatic schema drift detection
4. Migration assistant for updating models

---

## Expected Outcome

After implementation:

```
User Interface Flow:
1. Generate Features → Schema automatically created and saved
2. Train Model → Schema recorded, model registered
3. Predict → Schema loaded, compatibility validated, predictions generated
4. UI shows: "✅ Schema synchronized - 77 features, StandardScaler"

If Mismatch:
5. UI warns: "⚠️ Schema version mismatch detected
             Training schema v1.0, current v1.1
             Features: 77→85 (8 new features added)
             Retrain model for best results?"

Backend:
- models/model_manifest.json contains all model-schema mappings
- Each feature file has embedded FeatureSchema metadata
- Prediction uses exact same pipeline as training
```

---

**This plan is ready for review. Please provide feedback on architecture, approach, and priorities before implementation begins.**
