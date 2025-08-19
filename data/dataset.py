"""
Dataset loader and preprocessing for Free Spoken Digit Dataset (FSDD).
Handles audio loading, MFCC feature extraction, and variable-length normalization.
"""

import os
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings("ignore")


class FSSDDataset(Dataset):
    """
    Free Spoken Digit Dataset with MFCC preprocessing.
    
    Features:
    - Loads audio from Hugging Face datasets
    - Extracts MFCC features with configurable parameters
    - Handles variable-length sequences with padding/truncation
    - Caches processed features for faster training
    """
    
    def __init__(
        self, 
        split: str = "train",
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_length: int = 32,  # Maximum MFCC time frames
        sample_rate: int = 8000,
        cache_features: bool = True
    ):
        """
        Args:
            split: 'train' or 'test' 
            n_mfcc: Number of MFCC coefficients (13 is standard)
            n_fft: FFT window size for STFT
            hop_length: Hop length for STFT 
            max_length: Max time frames (for padding/truncation)
            sample_rate: Target sample rate for audio
            cache_features: Whether to cache MFCC features
        """
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.cache_features = cache_features
        
        # Load FSDD from Hugging Face
        print(f"Loading FSDD dataset ({split} split)...")
        dataset = load_dataset("mteb/free-spoken-digit-dataset", split=split)
        
        self.audio_data = []
        self.labels = []
        self.cached_mfccs = {} if cache_features else None
        
        # Process the dataset
        for i, item in enumerate(dataset):
            audio = item['audio']['array']
            label = item['label']
            
            self.audio_data.append(audio)
            self.labels.append(label)
            
        print(f"Loaded {len(self.audio_data)} samples")
        print(f"Label distribution: {np.bincount(self.labels)}")
        
    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Raw audio waveform
            
        Returns:
            mfcc: MFCC features of shape (n_mfcc, time_frames)
        """
        # Ensure audio is not empty and has minimum length
        if len(audio) < self.hop_length:
            audio = np.pad(audio, (0, self.hop_length - len(audio)))
            
        # Extract MFCCs using librosa (more robust than torchaudio for this)
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True
        )
        
        # Normalize MFCCs (mean=0, std=1 per coefficient)
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
        
        return mfcc
    
    def _normalize_length(self, mfcc: np.ndarray) -> np.ndarray:
        """
        Normalize MFCC length for batching.
        
        Strategy: Pad short sequences, truncate long ones to max_length.
        Alternative strategies could include:
        - Dynamic padding per batch
        - Interpolation/resampling
        - Attention masks for variable length
        
        Args:
            mfcc: MFCC of shape (n_mfcc, time_frames)
            
        Returns:
            normalized_mfcc: Shape (n_mfcc, max_length)
        """
        _, time_frames = mfcc.shape
        
        if time_frames < self.max_length:
            # Pad with zeros on the right
            pad_width = ((0, 0), (0, self.max_length - time_frames))
            mfcc = np.pad(mfcc, pad_width, mode='constant', constant_values=0)
        elif time_frames > self.max_length:
            # Truncate from the center to preserve beginning and end
            start = (time_frames - self.max_length) // 2
            mfcc = mfcc[:, start:start + self.max_length]
            
        return mfcc
    
    def __len__(self) -> int:
        return len(self.audio_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            mfcc: Tensor of shape (n_mfcc, max_length)
            label: Tensor of shape () with digit class (0-9)
        """
        # Check cache first
        if self.cache_features and idx in self.cached_mfccs:
            mfcc = self.cached_mfccs[idx]
        else:
            # Extract and normalize MFCC
            audio = self.audio_data[idx]
            mfcc = self._extract_mfcc(audio)
            mfcc = self._normalize_length(mfcc)
            
            # Cache if enabled
            if self.cache_features:
                self.cached_mfccs[idx] = mfcc
        
        # Convert to tensors
        mfcc_tensor = torch.FloatTensor(mfcc)
        label_tensor = torch.LongTensor([self.labels[idx]])
        
        return mfcc_tensor, label_tensor.squeeze()


def create_data_loaders(
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        batch_size: Batch size for training
        val_split: Fraction of train data to use for validation
        num_workers: Number of worker processes for data loading
        **dataset_kwargs: Arguments passed to FSSDDataset
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Load full training set
    full_train_dataset = FSSDDataset(split="train", **dataset_kwargs)
    
    # Split into train and validation
    train_indices, val_indices = train_test_split(
        range(len(full_train_dataset)),
        test_size=val_split,
        stratify=full_train_dataset.labels,
        random_state=42
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)
    
    # Test dataset
    test_dataset = FSSDDataset(split="test", **dataset_kwargs)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Data splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test of the dataset
    print("Testing FSDD dataset...")
    
    dataset = FSSDDataset(split="train", max_length=32)
    print(f"Dataset size: {len(dataset)}")
    
    # Test a sample
    mfcc, label = dataset[0]
    print(f"MFCC shape: {mfcc.shape}")  # Should be (13, 32)
    print(f"Label: {label}")
    print(f"MFCC range: [{mfcc.min():.3f}, {mfcc.max():.3f}]")
    
    # Test data loaders
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=8)
    
    batch_mfcc, batch_labels = next(iter(train_loader))
    print(f"Batch MFCC shape: {batch_mfcc.shape}")  # Should be (8, 13, 32)
    print(f"Batch labels shape: {batch_labels.shape}")  # Should be (8,)
