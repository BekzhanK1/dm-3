import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score

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

# Phase 1: Screening Space (OAT)
SCREENING_SPACE = {
    "embedding_dim": [64, 128, 256],
    "lstm_units": [64, 128, 256, 512],
    "dropout_rate": [0.2, 0.3, 0.5],
    "num_lstm_layers": [1, 2, 3],
}

def train_and_evaluate(config: Dict[str, Any], X_train, y_train, X_val, y_val, verbose=0, epochs=3) -> Tuple[float, float, float]:
    """
    Trains a model and returns (train_acc, val_acc, val_f1).
    """
    # Extract training-specific params
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
        epochs=epochs, 
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=verbose,
    )
    
    # Get best epoch metrics
    val_acc_history = history.history["val_accuracy"]
    train_acc_history = history.history["accuracy"]
    best_epoch_idx = np.argmax(val_acc_history)
    
    val_acc = val_acc_history[best_epoch_idx]
    train_acc = train_acc_history[best_epoch_idx]
    
    # Calculate F1 on validation set
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    val_f1 = f1_score(y_val, y_pred)
    
    return train_acc, val_acc, val_f1

def run_screening_phase(X_train, y_train, X_val, y_val) -> List[Dict]:
    """
    Phase 1: Fast Importance Screening (Low-Fidelity).
    Uses 30% data (passed in), 3 epochs.
    Ranks by Delta F1.
    """
    print("\n>>> PHASE 1: Fast Importance Screening (Low-Fidelity)")
    print("Running Baseline Model...")
    _, base_val_acc, base_val_f1 = train_and_evaluate(BASELINE_CONFIG, X_train, y_train, X_val, y_val, verbose=1, epochs=3)
    print(f"Baseline Results - Val Acc: {base_val_acc:.4f}, Val F1: {base_val_f1:.4f}")
    
    results = []
    total_params = len(SCREENING_SPACE)
    
    for idx, (param, values) in enumerate(SCREENING_SPACE.items(), 1):
        print(f"\n[Screening] ({idx}/{total_params}) Testing sensitivity for: '{param}'")
        best_param_delta_f1 = 0
        
        for v_idx, value in enumerate(values, 1):
            # Skip if value is same as baseline to save time, but for plotting we might want it.
            # Let's run it if it's not baseline, or just use baseline result.
            if value == BASELINE_CONFIG.get(param):
                continue
                
            config = BASELINE_CONFIG.copy()
            config[param] = value
            
            print(f"  -> [{v_idx}/{len(values)}] Setting {param} = {value}")
            _, val_acc, val_f1 = train_and_evaluate(config, X_train, y_train, X_val, y_val, epochs=3)
            
            delta_f1 = val_f1 - base_val_f1
            print(f"     Result: Val F1 = {val_f1:.4f} (Delta: {delta_f1:+.4f})")
            
            if abs(delta_f1) > abs(best_param_delta_f1):
                best_param_delta_f1 = delta_f1
        
        results.append({
            "parameter": param,
            "max_delta_f1": best_param_delta_f1,
            "abs_impact": abs(best_param_delta_f1)
        })
        
    # Rank by absolute impact on F1
    results.sort(key=lambda x: x["abs_impact"], reverse=True)
    return results

def run_local_search_phase(top_k_params: List[str], X_train, y_train, X_val, y_val) -> Tuple[Dict[str, Any], Dict[str, List[Tuple[Any, float]]]]:
    """
    Phase 2: Local Optimum Search (High-Fidelity, Adaptive).
    Uses full data.
    Returns: (best_config, search_history)
    """
    print("\n>>> PHASE 2: Local Optimum Search (High-Fidelity)")
    
    best_config = BASELINE_CONFIG.copy()
    search_history = {} # Stores results for plotting: param -> [(val, f1), ...]
    
    # We tune one parameter at a time, starting from the most important
    for param in top_k_params:
        print(f"\n[Local Search] Tuning parameter: {param}")
        
        # 1. Coarse Sweep
        # Define coarse range based on parameter type
        if param == "lstm_units":
            coarse_values = [64, 128, 256, 512]
        elif param == "embedding_dim":
            coarse_values = [64, 128, 256]
        elif param == "dropout_rate":
            coarse_values = [0.2, 0.3, 0.4, 0.5]
        elif param == "learning_rate":
            coarse_values = [0.0001, 0.001, 0.01]
        elif param == "batch_size":
            coarse_values = [32, 64, 128]
        elif param == "num_lstm_layers":
            coarse_values = [1, 2, 3]
        else:
            coarse_values = SCREENING_SPACE.get(param, [])
            
        print(f"  Coarse Sweep: {coarse_values}")
        
        best_val = -1
        best_f1 = -1
        results = []
        
        for val in coarse_values:
            config = best_config.copy()
            config[param] = val
            _, _, f1 = train_and_evaluate(config, X_train, y_train, X_val, y_val, epochs=3)
            results.append((val, f1))
            print(f"    {param}={val} -> F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_val = val
                
        # 1.5 Edge Expansion (Check if best value is at boundary)
        # Sort results to easily check edges
        sorted_results = sorted(results, key=lambda x: x[0])
        min_val = sorted_results[0][0]
        max_val = sorted_results[-1][0]
        
        expansion_attempts = 0
        max_expansions = 2 # Limit expansions to avoid infinite loops
        
        while expansion_attempts < max_expansions:
            expanded_val = None
            
            if best_val == max_val:
                # Try larger
                if param in ["lstm_units", "embedding_dim", "batch_size"]:
                    expanded_val = best_val * 2
                elif param == "num_lstm_layers":
                    expanded_val = best_val + 1
                elif param == "dropout_rate":
                    expanded_val = round(best_val + 0.1, 2)
                    if expanded_val >= 1.0: expanded_val = None
                elif param == "learning_rate":
                    expanded_val = best_val * 10
                    
            elif best_val == min_val:
                # Try smaller
                if param in ["lstm_units", "embedding_dim", "batch_size"]:
                    expanded_val = int(best_val / 2)
                    if expanded_val < 1: expanded_val = None
                elif param == "num_lstm_layers":
                    expanded_val = best_val - 1
                    if expanded_val < 1: expanded_val = None
                elif param == "dropout_rate":
                    expanded_val = round(best_val - 0.1, 2)
                    if expanded_val < 0: expanded_val = None
                elif param == "learning_rate":
                    expanded_val = best_val / 10
            
            # If no valid expansion or already tested, break
            if expanded_val is None or expanded_val in [r[0] for r in results]:
                break
                
            print(f"  ! Best value ({best_val}) is at edge. Expanding search to {expanded_val}...")
            config = best_config.copy()
            config[param] = expanded_val
            _, _, f1 = train_and_evaluate(config, X_train, y_train, X_val, y_val, epochs=3)
            results.append((expanded_val, f1))
            print(f"    {param}={expanded_val} -> F1: {f1:.4f}")
            
            # Update bounds and best
            if f1 > best_f1:
                best_f1 = f1
                best_val = expanded_val
                # Update min/max for next iteration
                if expanded_val > max_val: max_val = expanded_val
                if expanded_val < min_val: min_val = expanded_val
            else:
                # Expansion didn't help, stop expanding
                break
                
            expansion_attempts += 1

        # 2. Zoom In (Fine Sweep) if possible
        # Re-sort results after expansion
        sorted_results = sorted(results, key=lambda x: x[0])
        best_idx = -1
        for i, (v, f) in enumerate(sorted_results):
            if v == best_val:
                best_idx = i
                break
        
        # Logic to zoom in
        fine_values = []
        if param in ["lstm_units", "embedding_dim", "batch_size"]:
            # Geometric mean neighbors
            if best_idx > 0:
                prev_val = sorted_results[best_idx-1][0]
                mid_left = int((prev_val + best_val) / 2)
                fine_values.append(mid_left)
            if best_idx < len(sorted_results) - 1:
                next_val = sorted_results[best_idx+1][0]
                mid_right = int((best_val + next_val) / 2)
                fine_values.append(mid_right)
                
        elif param == "dropout_rate":
            # +/- 0.05
            fine_values = [round(best_val - 0.05, 2), round(best_val + 0.05, 2)]
            fine_values = [v for v in fine_values if 0 <= v <= 0.9]
            
        if fine_values:
            print(f"  Fine Sweep (Zooming in): {fine_values}")
            for val in fine_values:
                if val in [r[0] for r in results]: continue # Skip if already tested
                
                config = best_config.copy()
                config[param] = val
                _, _, f1 = train_and_evaluate(config, X_train, y_train, X_val, y_val, epochs=3)
                results.append((val, f1))
                print(f"    {param}={val} -> F1: {f1:.4f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_val = val
        
        # Update best config for next parameter
        best_config[param] = best_val
        search_history[param] = sorted(results, key=lambda x: x[0]) # Store sorted results
        print(f"  -> Selected optimal {param} = {best_val} (F1: {best_f1:.4f})")
        
    return best_config, search_history
