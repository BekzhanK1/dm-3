import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping

from preprocess_fake_news import get_data, VOCAB_SIZE, MAX_LENGTH
from model import build_lstm_model, build_transformer_model
import tuner

import tensorflow as tf

# Configuration
MODEL_TYPE = "lstm" # Options: "lstm", "transformer"

def create_plot_dirs():
    """Creates directory structure for saving plots."""
    dirs = ["plots/phase1_screening", "plots/phase2_local_search", "plots/final_eval"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Created plot directories: plots/")

def check_gpu():
    """Checks and prints available GPUs."""
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n[System Check] Num GPUs Available: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            print(f"  - Found GPU: {gpu}")
    else:
        print("  ! WARNING: No GPU detected. Training will be slow.")

def plot_history(history, title="Model Performance", save_path="plots/final_eval/training_curves.png"):
    """Plots training and validation accuracy/loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training curves to {save_path}")
    plt.close()

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test set and prints metrics."""
    print("\n=== Final Evaluation on Test Set ===")
    
    # Predict
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'True']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('plots/final_eval/confusion_matrix.png')
    print("Saved confusion matrix to plots/final_eval/confusion_matrix.png")

def plot_param_importance(results):
    """Plots a bar chart of hyperparameter importance (Delta F1)."""
    params = [res['parameter'] for res in results]
    impacts = [res['abs_impact'] for res in results]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=impacts, y=params, palette='viridis')
    plt.title('Hyperparameter Importance (Sensitivity Analysis)')
    plt.xlabel('Absolute Delta F1-Score')
    plt.ylabel('Hyperparameter')
    plt.tight_layout()
    plt.savefig('plots/phase1_screening/param_importance.png')
    print("Saved importance plot to plots/phase1_screening/param_importance.png")

def plot_local_search_results(search_history):
    """
    Plots the results of the local search phase (Param Value vs F1 Score).
    Identifies the 'sweet spot'.
    """
    print("Generating local search plots...")
    for param, results in search_history.items():
        # results is a list of (value, f1) tuples, already sorted by value
        values = [r[0] for r in results]
        f1_scores = [r[1] for r in results]
        
        plt.figure(figsize=(8, 5))
        
        # Handle categorical/string values if any (though mostly numerical here)
        if isinstance(values[0], str) or isinstance(values[0], bool):
             plt.plot([str(v) for v in values], f1_scores, 'bo-', marker='o')
             max_x = str(values[f1_scores.index(max(f1_scores))])
        else:
             plt.plot(values, f1_scores, 'bo-', marker='o')
             max_x = values[f1_scores.index(max(f1_scores))]
             
        plt.title(f'Local Search: {param} vs F1 Score')
        plt.xlabel(f'{param} Value')
        plt.ylabel('Validation F1 Score')
        plt.grid(True)
        
        # Highlight max point (Sweet Spot)
        max_f1 = max(f1_scores)
        plt.plot(max_x, max_f1, 'r*', markersize=15, label=f'Sweet Spot: {max_f1:.4f}')
        plt.legend()
        
        save_path = f"plots/phase2_local_search/{param}_optimization.png"
        plt.savefig(save_path)
        plt.close()
        print(f"  Saved plot to {save_path}")

def main():
    create_plot_dirs()
    check_gpu()
    
    print("===========================================================")
    print(f"   Fake News Detection Pipeline (Model: {MODEL_TYPE.upper()})")
    print("===========================================================")

    # ---------------------------------------------------------
    # PHASE 1: Fast Importance Screening (Low-Fidelity)
    # ---------------------------------------------------------
    print("\n>>> PHASE 1: Data Loading (30% Subset)")
    X_train_small, y_train_small, X_val_small, y_val_small, _, _, _ = get_data(subset_fraction=0.3)
    
    print("\n>>> PHASE 1: Running Importance Screening...")
    screening_results = tuner.run_screening_phase(X_train_small, y_train_small, X_val_small, y_val_small, model_type=MODEL_TYPE)
    
    plot_param_importance(screening_results)
    
    print("\n[Phase 1 Result] Hyperparameter Sensitivity Ranking (by Delta F1):")
    for i, res in enumerate(screening_results):
        print(f"  {i+1}. {res['parameter']:<20} (Impact: {res['abs_impact']:.4f})")
        
    # Select Top-K (e.g., k=2)
    k = 2
    top_k_params = [res["parameter"] for res in screening_results[:k]]
    print(f"\n[Decision] Selected Top-{k} Parameters for Local Search: {top_k_params}")
    
    # ---------------------------------------------------------
    # PHASE 2: Local Optimum Search (High-Fidelity)
    # ---------------------------------------------------------
    print("\n>>> PHASE 2: Data Loading (Full Dataset)")
    X_train, y_train, X_val, y_val, X_test, y_test, tokenizer = get_data(subset_fraction=1.0)
    
    print("\n>>> PHASE 2: Running Local Optimum Search...")
    best_config, search_history = tuner.run_local_search_phase(top_k_params, X_train, y_train, X_val, y_val, model_type=MODEL_TYPE)
    
    plot_local_search_results(search_history)
    
    print("\n[Phase 2 Result] Best Configuration Found:")
    for param, value in best_config.items():
        print(f"  {param}: {value}")
        
    # ---------------------------------------------------------
    # PHASE 3: Final Model Training
    # ---------------------------------------------------------
    print("\n>>> PHASE 3: Training Final Model")
    print("Retraining model with best configuration on full training set...")
    
    # Re-build model with best config
    model_params = {k: v for k, v in best_config.items() if k != "batch_size"}
    model_params["vocab_size"] = VOCAB_SIZE
    model_params["input_length"] = MAX_LENGTH
    
    if MODEL_TYPE == "lstm":
        final_model = build_lstm_model(**model_params)
    elif MODEL_TYPE == "transformer":
        final_model = build_transformer_model(**model_params)
    
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    
    batch_size = best_config.get("batch_size", 64)
    
    history = final_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=15, # Sufficient epochs as requested
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )
    
    # ---------------------------------------------------------
    # Final Evaluation
    # ---------------------------------------------------------
    plot_history(history)
    evaluate_model(final_model, X_test, y_test)
    
    # Save Model
    save_name = f"best_fake_news_{MODEL_TYPE}_model.keras"
    final_model.save(save_name)
    print("\n===========================================================")
    print("   Pipeline Completed Successfully")
    print(f"   Model saved to: {save_name}")
    print("   Plots saved to: plots/")
    print("===========================================================")

if __name__ == "__main__":
    main()
