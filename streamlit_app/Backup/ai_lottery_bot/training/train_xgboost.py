# train_xgboost.py

def train_xgboost(features, labels, **kwargs):
    """Train an XGBoost model if available; fallback to RandomForestRegressor.

    Returns the trained model object.
    
    This is a legacy function. For advanced XGBoost training with comprehensive
    features, use the advanced_xgboost module instead.
    """
    # Try to use the advanced XGBoost trainer
    try:
        from ai_lottery_bot.training.advanced_xgboost import train_advanced_xgboost
        import numpy as np
        
        # Convert to numpy arrays if needed
        X = np.array(features) if not isinstance(features, np.ndarray) else features
        y = np.array(labels) if not isinstance(labels, np.ndarray) else labels
        
        # Extract relevant parameters from kwargs
        hyperparams = kwargs.get('hyperparams', {})
        save_dir = kwargs.get('save_dir', None)
        
        # Train using advanced method
        metadata = train_advanced_xgboost(
            X=X,
            y=y,
            hyperparams=hyperparams,
            save_dir=save_dir,
            validation_split=0.2
        )
        
        # For backward compatibility, return the model path
        import joblib
        model = joblib.load(metadata['file'])
        return model
        
    except Exception as e:
        print(f"Advanced XGBoost training failed: {e}")
        print("Falling back to basic training...")
        
        # Original fallback implementation
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42
            )
        except Exception:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                random_state=42
            )

        model.fit(features, labels)
        return model
