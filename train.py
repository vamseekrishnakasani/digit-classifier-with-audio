#!/usr/bin/env python3
"""
Training Script for Spoken Digit Classification
Simple, focused training for the 2D CNN model.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm

# Add module paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

from data.dataset import create_data_loaders
from model.models import get_model, count_parameters


class SimpleTrainer:
    """Simple, focused trainer for digit classification."""
    
    def __init__(self, model, train_loader, val_loader, device='cpu', lr=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training setup
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3)
        
        # Tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(tqdm(self.train_loader, desc="Training")):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc="Validating"):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def train(self, epochs):
        """Main training loop."""
        print(f"🚀 Starting training for {epochs} epochs")
        print("-" * 50)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train and validate
            train_loss = self.train_epoch()
            val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Track best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc)
                best_marker = " ⭐ NEW BEST!"
            else:
                best_marker = ""
            
            # Logging
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_acc:5.2f}% | "
                  f"Time: {epoch_time:.1f}s{best_marker}")
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
        
        print(f"\n✅ Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def save_checkpoint(self, epoch, val_acc):
        """Save model checkpoint."""
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies
        }
        torch.save(checkpoint, 'checkpoints/best_model.pth')


def quick_test(model, test_loader, device):
    """Quick test on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Testing"):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def main():
    """Main training function."""
    print("🎤 Spoken Digit Classifier Training")
    print("=" * 40)
    
    # Configuration
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Load data
    print("\n📊 Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=BATCH_SIZE, num_workers=2
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\n🧠 Creating CNN model...")
    model = get_model()
    
    # Create trainer
    trainer = SimpleTrainer(model, train_loader, val_loader, device, LEARNING_RATE)
    
    # Train
    trainer.train(EPOCHS)
    
    # Quick test
    test_accuracy = quick_test(model, test_loader, device)
    
    # Final summary
    print(f"\n📋 Training Summary")
    print("-" * 25)
    print(f"Model: CNN")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Best Val Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Target (95%): {'✅ ACHIEVED' if test_accuracy >= 95 else '❌ RETRY WITH MORE EPOCHS'}")


if __name__ == "__main__":
    main()
