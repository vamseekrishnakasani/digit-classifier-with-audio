#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluate the trained model and generate performance metrics.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add module paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

from data.dataset import create_data_loaders
from model.models import get_model


def load_trained_model(model_path):
    """Load the trained model from checkpoint."""
    model = get_model()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded CNN model")
    print(f"   Training epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    return model


def evaluate_model(model, data_loader, dataset_name):
    """Evaluate model on dataset and return predictions."""
    all_predictions = []
    all_targets = []
    
    print(f"\n📊 Evaluating on {dataset_name}...")
    
    with torch.no_grad():
        for data, targets in data_loader:
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.numpy())
            all_targets.extend(targets.numpy())
    
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    accuracy = accuracy_score(targets, predictions)
    
    print(f"   Accuracy: {accuracy*100:.2f}% ({sum(predictions == targets)}/{len(targets)})")
    
    return targets, predictions, accuracy * 100


def plot_confusion_matrix(targets, predictions, dataset_name, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.colorbar()
    
    # Add labels
    plt.xticks(range(10), range(10))
    plt.yticks(range(10), range(10))
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    
    # Add numbers to cells
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(cm[i, j]), 
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Confusion matrix saved: {save_path}")
    
    plt.show()


def print_classification_report(targets, predictions, dataset_name):
    """Print detailed classification report."""
    print(f"\n📋 Classification Report - {dataset_name}")
    print("=" * 50)
    print(classification_report(targets, predictions, 
                              target_names=[f"Digit {i}" for i in range(10)]))


def analyze_errors(targets, predictions, dataset_name):
    """Analyze common misclassifications."""
    cm = confusion_matrix(targets, predictions)
    
    print(f"\n🔍 Error Analysis - {dataset_name}")
    print("=" * 40)
    
    # Find most common errors
    errors = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                errors.append((i, j, cm[i, j]))
    
    # Sort by frequency
    errors.sort(key=lambda x: x[2], reverse=True)
    
    if errors:
        print("Most common misclassifications:")
        for true_digit, pred_digit, count in errors[:5]:
            percentage = count / cm[true_digit].sum() * 100
            print(f"   {true_digit} → {pred_digit}: {count} errors ({percentage:.1f}%)")
    else:
        print("No misclassifications found!")


def main():
    """Main evaluation function."""
    print("🎯 Model Evaluation")
    print("=" * 40)
    
    # Configuration
    MODEL_PATH = "checkpoints/best_model.pth"
    BATCH_SIZE = 32
    
    # Load data
    print("📊 Loading datasets...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=BATCH_SIZE, num_workers=2
    )
    
    # Load model
    model = load_trained_model(MODEL_PATH)
    
    # Evaluate on validation set
    val_targets, val_predictions, val_accuracy = evaluate_model(
        model, val_loader, "Validation Set"
    )
    
    # Evaluate on test set
    test_targets, test_predictions, test_accuracy = evaluate_model(
        model, test_loader, "Test Set"
    )
    
    # Generate reports
    print_classification_report(val_targets, val_predictions, "Validation Set")
    print_classification_report(test_targets, test_predictions, "Test Set")
    
    # Error analysis
    analyze_errors(val_targets, val_predictions, "Validation Set")
    analyze_errors(test_targets, test_predictions, "Test Set")
    
    # Create plots
    os.makedirs('evaluation_plots', exist_ok=True)
    plot_confusion_matrix(val_targets, val_predictions, "Validation Set", 
                         'evaluation_plots/confusion_matrix_validation.png')
    plot_confusion_matrix(test_targets, test_predictions, "Test Set",
                         'evaluation_plots/confusion_matrix_test.png')
    
    # Final summary
    print(f"\n🎯 Final Summary")
    print("=" * 30)
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Target (95%): {'✅ ACHIEVED' if test_accuracy >= 95 else '❌ NEEDS IMPROVEMENT'}")
    print(f"📁 Plots saved in: evaluation_plots/")


if __name__ == "__main__":
    main()
