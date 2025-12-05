"""
Prediction Generation Tracing Service
Logs detailed step-by-step information about the prediction generation process
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from io import StringIO

class PredictionTracer:
    """Traces prediction generation and logs detailed steps."""
    
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
    def start(self, game: str, model_type: str, count: int, mode: str):
        """Start prediction tracing."""
        self.start_time = datetime.now()
        self.logs = []
        self.log("STARTED", f"Generating {count} predictions for {game} using {model_type} ({mode} mode)")
        self.log("CONFIG", f"Game: {game}, Model: {model_type}, Mode: {mode}, Count: {count}")
    
    def log(self, category: str, message: str, level: str = "INFO", data: Optional[Dict] = None):
        """Log a trace entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "message": message,
            "level": level,
        }
        if data:
            entry["data"] = data
        self.logs.append(entry)
    
    def end(self):
        """End prediction tracing."""
        self.end_time = datetime.now()
        if self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
            self.log("COMPLETED", f"Prediction generation completed in {duration:.2f}s")
    
    def log_feature_loading(self, model_type: str, features_shape: tuple, success: bool, error: Optional[str] = None):
        """Log feature loading step."""
        if success:
            self.log("FEATURE_LOAD", f"✅ Loaded {model_type} features", data={"shape": features_shape})
        else:
            self.log("FEATURE_LOAD", f"❌ Failed to load {model_type} features: {error}", level="WARNING", data={"error": error})
    
    def log_model_loading(self, model_type: str, model_path: str, success: bool, error: Optional[str] = None):
        """Log model loading step."""
        if success:
            self.log("MODEL_LOAD", f"✅ Loaded {model_type} model from {model_path}")
        else:
            self.log("MODEL_LOAD", f"❌ Failed to load {model_type} model: {error}", level="ERROR", data={"error": error})
    
    def log_scaler_info(self, scaler_type: str, features_count: int):
        """Log scaler information."""
        self.log("SCALER", f"Using scaler with {features_count} features", data={"scaler_type": scaler_type})
    
    def log_prediction_attempt(self, iteration: int, input_shape: tuple, output_shape: tuple, confidence: float):
        """Log individual prediction attempt."""
        self.log("PREDICTION", f"Set {iteration}: input_shape={input_shape}, output_shape={output_shape}, confidence={confidence:.2%}")
    
    def log_model_output(self, iteration: int, pred_probs_shape: tuple, num_classes: int, top_probs: List[float]):
        """Log model output details."""
        self.log("MODEL_OUTPUT", f"Set {iteration}: {num_classes} classes detected", data={
            "shape": pred_probs_shape,
            "top_probs": [float(p) for p in top_probs[:5]]
        })
    
    def log_number_selection(self, iteration: int, numbers: List[int], method: str, confidence: float):
        """Log number selection process."""
        self.log("NUMBER_SELECT", f"Set {iteration}: {numbers} selected via {method}", data={
            "numbers": numbers,
            "method": method,
            "confidence": confidence
        })
    
    def log_fallback(self, iteration: int, reason: str, fallback_type: str):
        """Log fallback event."""
        self.log("FALLBACK", f"Set {iteration}: Using {fallback_type} - Reason: {reason}", level="WARNING")
    
    def log_error(self, error_msg: str, exception_details: Optional[str] = None):
        """Log error."""
        self.log("ERROR", error_msg, level="ERROR", data={"exception": exception_details} if exception_details else None)
    
    # ===== Detailed Step Logging Methods =====
    
    def log_data_loading(self, data_type: str, source: str, success: bool, details: Optional[Dict] = None):
        """Log data loading step."""
        if success:
            self.log("DATA_LOAD", f"✅ Loaded {data_type} from {source}", data=details or {})
        else:
            self.log("DATA_LOAD", f"❌ Failed to load {data_type}: {source}", level="ERROR", data=details or {})
    
    def log_config_load(self, game: str, config_data: Dict):
        """Log configuration loading."""
        self.log("CONFIG_LOAD", f"✅ Loaded config for {game}", data={
            "main_numbers": config_data.get("main_numbers"),
            "bonus_numbers": config_data.get("bonus_numbers"),
            "max_number": config_data.get("max_number")
        })
    
    def log_model_search(self, game: str, model_type: str, found: bool, model_info: Optional[Dict] = None):
        """Log model search/discovery."""
        if found:
            self.log("MODEL_SEARCH", f"✅ Found {model_type} model for {game}", data=model_info or {})
        else:
            self.log("MODEL_SEARCH", f"⚠️ No {model_type} model found for {game}", level="WARNING")
    
    def log_model_load_start(self, model_path: str):
        """Log start of model loading."""
        self.log("MODEL_LOAD_START", f"🔄 Loading model from: {model_path}")
    
    def log_model_load_complete(self, model_path: str, file_size_mb: float, load_time_ms: float):
        """Log successful model load completion."""
        self.log("MODEL_LOAD_COMPLETE", f"✅ Model loaded successfully", data={
            "path": model_path,
            "file_size_mb": f"{file_size_mb:.2f}",
            "load_time_ms": f"{load_time_ms:.2f}"
        })
    
    def log_feature_generation(self, iteration: int, feature_count: int, method: str):
        """Log feature generation for each set."""
        self.log("FEATURE_GEN", f"Set {iteration}: Generated {feature_count} features via {method}", data={
            "set": iteration,
            "features": feature_count,
            "method": method
        })
    
    def log_feature_normalization(self, iteration: int, scaler_type: str, input_shape: tuple, output_shape: tuple):
        """Log feature normalization step."""
        self.log("FEATURE_NORM", f"Set {iteration}: Normalized with {scaler_type}", data={
            "set": iteration,
            "input_shape": str(input_shape),
            "output_shape": str(output_shape),
            "scaler": scaler_type
        })
    
    def log_model_prediction_start(self, iteration: int, model_type: str, input_shape: tuple):
        """Log start of model prediction."""
        self.log("PREDICT_START", f"Set {iteration}: Running {model_type} prediction with input {input_shape}")
    
    def log_model_prediction_output(self, iteration: int, model_type: str, output_shape: tuple, 
                                   raw_output_sample: Optional[List] = None):
        """Log model prediction output details."""
        data = {
            "set": iteration,
            "model": model_type,
            "output_shape": str(output_shape)
        }
        if raw_output_sample:
            data["sample_output"] = [f"{v:.4f}" for v in raw_output_sample[:10]]
        self.log("PREDICT_OUTPUT", f"Set {iteration}: {model_type} output shape {output_shape}", data=data)
    
    def log_number_extraction(self, iteration: int, raw_numbers: List[int], filtered_numbers: List[int], 
                             method: str, details: Optional[Dict] = None):
        """Log number extraction and filtering process."""
        self.log("NUMBER_EXTRACT", f"Set {iteration}: {method} - Selected {len(filtered_numbers)} from {len(raw_numbers)} candidates", data={
            "set": iteration,
            "raw_count": len(raw_numbers),
            "raw_numbers": raw_numbers[:20],  # Show first 20
            "final_count": len(filtered_numbers),
            "final_numbers": filtered_numbers,
            "method": method,
            **(details or {})
        })
    
    def log_confidence_calculation(self, iteration: int, confidence: float, method: str, 
                                  raw_scores: Optional[List] = None):
        """Log confidence score calculation."""
        data = {
            "set": iteration,
            "confidence": f"{confidence:.4f}",
            "percentage": f"{confidence*100:.2f}%",
            "method": method
        }
        if raw_scores:
            data["raw_scores_sample"] = [f"{s:.4f}" for s in raw_scores[:5]]
        self.log("CONFIDENCE", f"Set {iteration}: Confidence = {confidence:.2%} via {method}", data=data)
    
    def log_validation_check(self, iteration: int, check_name: str, passed: bool, details: Optional[Dict] = None):
        """Log validation check results."""
        status = "✅" if passed else "❌"
        level = "INFO" if passed else "WARNING"
        self.log("VALIDATION", f"Set {iteration}: {status} {check_name}", level=level, data={
            "set": iteration,
            "check": check_name,
            "passed": passed,
            **(details or {})
        })
    
    def log_ensemble_voting(self, iteration: int, model_votes: Dict[str, List[int]], 
                           final_numbers: List[int], vote_distribution: Optional[Dict] = None):
        """Log ensemble voting results."""
        self.log("ENSEMBLE_VOTE", f"Set {iteration}: Ensemble selected {final_numbers}", data={
            "set": iteration,
            "model_votes": model_votes,
            "final_selection": final_numbers,
            "vote_distribution": vote_distribution or {}
        })
    
    def log_final_set(self, iteration: int, numbers: List[int], confidence: float, metadata: Optional[Dict] = None):
        """Log final prediction set."""
        self.log("FINAL_SET", f"✅ Set {iteration}: {sorted(numbers)} (confidence: {confidence:.2%})", data={
            "set": iteration,
            "numbers": sorted(numbers),
            "confidence": f"{confidence:.4f}",
            "count": len(numbers),
            **(metadata or {})
        })
    
    def log_batch_complete(self, total_sets: int, avg_confidence: float, errors_encountered: int):
        """Log batch completion."""
        self.log("BATCH_COMPLETE", f"✅ Generated {total_sets} prediction sets (avg confidence: {avg_confidence:.2%}, errors: {errors_encountered})", data={
            "total_sets": total_sets,
            "avg_confidence": f"{avg_confidence:.4f}",
            "errors": errors_encountered
        })
    
    def get_formatted_logs(self) -> str:
        """Get formatted log output for display."""
        output = []
        output.append("=" * 100)
        output.append("PREDICTION GENERATION LOG - DETAILED TRACE")
        output.append("=" * 100)
        
        for idx, entry in enumerate(self.logs):
            timestamp = entry.get("timestamp", "")[11:19]  # HH:MM:SS
            category = entry.get("category", "").ljust(18)
            level = entry.get("level", "INFO")
            message = entry.get("message", "")
            
            # Format based on level
            if level == "ERROR":
                prefix = "❌"
            elif level == "WARNING":
                prefix = "⚠️ "
            elif level == "DEBUG":
                prefix = "🔍"
            else:
                prefix = "ℹ️ "
            
            line = f"{prefix} [{timestamp}] {category} │ {message}"
            output.append(line)
            
            # Add data if present - more detailed formatting
            if "data" in entry:
                data = entry["data"]
                keys = list(data.keys())
                for key_idx, key in enumerate(keys):
                    val = data[key]
                    is_last = key_idx == len(keys) - 1
                    connector = "└─" if is_last else "├─"
                    
                    # Format value based on type
                    if isinstance(val, dict):
                        output.append(f"        {connector} {key}:")
                        for sub_key, sub_val in val.items():
                            output.append(f"        │   └─ {sub_key}: {sub_val}")
                    elif isinstance(val, list):
                        if len(val) > 5:
                            val_str = f"{val[:5]} ... ({len(val)} total)"
                        else:
                            val_str = str(val)
                        output.append(f"        {connector} {key}: {val_str}")
                    else:
                        output.append(f"        {connector} {key}: {val}")
        
        output.append("=" * 100)
        return "\n".join(output)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_logs": len(self.logs),
            "errors": len([l for l in self.logs if l.get("level") == "ERROR"]),
            "warnings": len([l for l in self.logs if l.get("level") == "WARNING"]),
            "predictions_logged": len([l for l in self.logs if l.get("category") == "PREDICTION"]),
            "fallbacks": len([l for l in self.logs if l.get("category") == "FALLBACK"]),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

# Global tracer instance
_prediction_tracer = PredictionTracer()

def get_prediction_tracer() -> PredictionTracer:
    """Get the global prediction tracer."""
    return _prediction_tracer

def reset_tracer():
    """Reset tracer for new prediction generation."""
    global _prediction_tracer
    _prediction_tracer = PredictionTracer()
