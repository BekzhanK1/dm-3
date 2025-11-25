import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

from preprocess_fake_news import get_data, VOCAB_SIZE, MAX_LENGTH
from model import build_lstm_model
import tuner

def plot_history(history, title="Model Performance"):
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
    plt.savefig('training_curves.png')
    print("Saved training curves to training_curves.png")
    # plt.show() # Commented out for non-interactive environments

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test set and prints metrics."""
    print("\n=== Final Evaluation on Test Set ===")
    
    # Predict
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'True']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix to confusion_matrix.png")

def main():
    print("===========================================================")
    print("   LSTM Fake News Detection Pipeline Started")
    print("===========================================================")

    # 1. Load Data
    print("\n>>> STEP 1: Data Loading & Preprocessing")
    X_train, y_train, X_val, y_val, X_test, y_test, tokenizer = get_data()
    print(f"Data loaded successfully.")
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # 2. Sensitivity Analysis
    print("\n>>> STEP 2: Empirical Hyperparameter Importance Analysis")
    print("Identifying the most critical hyperparameters...")
    sensitivity_results = tuner.run_sensitivity_analysis(X_train, y_train, X_val, y_val)
    
    print("\n[Analysis Result] Parameter Importance Ranking:")
    for i, res in enumerate(sensitivity_results):
        print(f"  {i+1}. {res['parameter']:<20} (Max Impact: {res['abs_impact']:.4f})")
        
    # Select Top-K (e.g., k=3)
    k = 3
    top_k_params = [res["parameter"] for res in sensitivity_results[:k]]
    print(f"\n[Decision] Selected Top-{k} Parameters for Tuning: {top_k_params}")
    
    # 3. Systematic Tuning
    print("\n>>> STEP 3: Systematic Hyperparameter Tuning (Grid Search)")
    print(f"Optimizing {top_k_params}...")
    best_config, best_val_acc = tuner.run_grid_search(top_k_params, X_train, y_train, X_val, y_val)
    
    print("\n[Tuning Result] Best Configuration Found:")
    for param, value in best_config.items():
        print(f"  {param}: {value}")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    
    # 4. Train Final Model with Best Config
    print("\n>>> STEP 4: Training Final Model")
    print("Retraining model with best configuration on full training set...")
    # Re-build model with best config
    model_params = {k: v for k, v in best_config.items() if k != "batch_size"}
    model_params["vocab_size"] = VOCAB_SIZE
    model_params["input_length"] = MAX_LENGTH
    
    final_model = build_lstm_model(**model_params)
    
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    
    batch_size = best_config.get("batch_size", 64)
    
    history = final_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10, # Train longer for final model
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )
    
    # 5. Evaluation
    print("\n>>> STEP 5: Final Evaluation")
    plot_history(history)
    evaluate_model(final_model, X_test, y_test)
    
    # Save Model
    final_model.save("best_fake_news_lstm_model.h5")
    print("\n===========================================================")
    print("   Pipeline Completed Successfully")
    print("   Model saved to: best_fake_news_lstm_model.h5")
    print("===========================================================")

if __name__ == "__main__":
    main()
