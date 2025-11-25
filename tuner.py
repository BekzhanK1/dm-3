import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from tensorflow.keras.callbacks import EarlyStopping

from preprocess_fake_news import get_data, VOCAB_SIZE, MAX_LENGTH
from model import build_lstm_model

# Baseline Hyperparameters
BASELINE_CONFIG = {
    "embedding_dim": 128,
    "lstm_units": 128,
    "num_lstm_layers": 1,
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 64,
    "optimizer_name": "Adam",
}

# Search Space for Sensitivity Analysis (One-at-a-Time)
SENSITIVITY_SPACE = {
    "embedding_dim": [64, 256],
    "lstm_units": [64, 256, 512],
    "num_lstm_layers": [2, 3],
    "dropout_rate": [0.1, 0.5],
    "learning_rate": [0.0005, 0.0001],
    "batch_size": [32, 128],
    "optimizer_name": ["RMSprop"],
}

def train_and_evaluate(config: Dict[str, Any], X_train, y_train, X_val, y_val, verbose=0) -> float:
    """
    Trains a model with the given config and returns validation accuracy.
    """
    # Extract training-specific params that are not model architecture params
    batch_size = config.get("batch_size", 64)
    
    # Filter config for model builder
    model_params = {k: v for k, v in config.items() if k != "batch_size"}
    model_params["vocab_size"] = VOCAB_SIZE
    model_params["input_length"] = MAX_LENGTH
    
    model = build_lstm_model(**model_params)
    
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=5, # Keep epochs low for tuning speed, or use early stopping
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=verbose,
    )
    
    val_acc = max(history.history["val_accuracy"])
    return val_acc

def run_sensitivity_analysis(X_train, y_train, X_val, y_val) -> List[Dict]:
    """
    Performs One-at-a-Time sensitivity analysis.
    """
    print("\n[Sensitivity Analysis] Starting Baseline Model training...")
    baseline_acc = train_and_evaluate(BASELINE_CONFIG, X_train, y_train, X_val, y_val, verbose=1)
    print(f"[Sensitivity Analysis] Baseline Validation Accuracy: {baseline_acc:.4f}")
    
    results = []
    total_params = len(SENSITIVITY_SPACE)
    
    for idx, (param, values) in enumerate(SENSITIVITY_SPACE.items(), 1):
        print(f"\n[Sensitivity Analysis] ({idx}/{total_params}) Testing sensitivity for parameter: '{param}'")
        best_param_delta = 0
        
        for v_idx, value in enumerate(values, 1):
            config = BASELINE_CONFIG.copy()
            config[param] = value
            
            print(f"  -> [{v_idx}/{len(values)}] Setting {param} = {value}")
            acc = train_and_evaluate(config, X_train, y_train, X_val, y_val)
            delta = acc - baseline_acc
            print(f"     Result: Val Acc = {acc:.4f} (Delta: {delta:+.4f})")
            
            # Track the maximum absolute impact of this parameter
            if abs(delta) > abs(best_param_delta):
                best_param_delta = delta
        
        results.append({
            "parameter": param,
            "max_delta": best_param_delta,
            "abs_impact": abs(best_param_delta)
        })
        
    # Sort by impact
    results.sort(key=lambda x: x["abs_impact"], reverse=True)
    return results

def run_grid_search(top_k_params: List[str], X_train, y_train, X_val, y_val) -> Tuple[Dict, float]:
    """
    Runs grid search on the top-k most important parameters.
    """
    print(f"\n[Grid Search] Starting Grid Search on Top-{len(top_k_params)} Parameters: {top_k_params}")
    
    # Create grid
    param_grid = {}
    for param in top_k_params:
        # Include baseline value + sensitivity values
        values = [BASELINE_CONFIG[param]] + SENSITIVITY_SPACE[param]
        param_grid[param] = sorted(list(set(values))) # Unique values
        
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    
    print(f"[Grid Search] Total hyperparameter combinations to test: {len(combinations)}")
    
    best_acc = 0.0
    best_config = BASELINE_CONFIG.copy()
    
    for i, values in enumerate(combinations):
        current_config = BASELINE_CONFIG.copy()
        combo_dict = dict(zip(keys, values))
        current_config.update(combo_dict)
        
        print(f"\n[Grid Search] Run {i+1}/{len(combinations)}")
        print(f"  Configuration: {combo_dict}")
        acc = train_and_evaluate(current_config, X_train, y_train, X_val, y_val)
        print(f"  -> Validation Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_config = current_config
            print(f"  -> New Best Model Found! (Acc: {best_acc:.4f})")
            
    return best_config, best_acc

def main():
    X_train, y_train, X_val, y_val, _, _, _ = get_data()
    
    # 1. Sensitivity Analysis
    print("=== Phase 1: Empirical Hyperparameter Importance Analysis ===")
    sensitivity_results = run_sensitivity_analysis(X_train, y_train, X_val, y_val)
    
    print("\nParameter Importance Ranking:")
    for i, res in enumerate(sensitivity_results):
        print(f"{i+1}. {res['parameter']} (Impact: {res['abs_impact']:.4f})")
        
    # Select Top-K (e.g., k=3)
    k = 3
    top_k_params = [res["parameter"] for res in sensitivity_results[:k]]
    print(f"\nSelected Top-{k} Parameters for Tuning: {top_k_params}")
    
    # 2. Systematic Tuning
    print("\n=== Phase 2: Systematic Hyperparameter Tuning ===")
    best_config, best_acc = run_grid_search(top_k_params, X_train, y_train, X_val, y_val)
    
    print("\n=== Tuning Complete ===")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print("Best Configuration:")
    print(best_config)

if __name__ == "__main__":
    main()
