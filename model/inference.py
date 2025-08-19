"""
Fast inference script for spoken digit classification.
Supports both file input and real-time microphone input.
"""

import torch
import torch.nn.functional as F
import librosa
import numpy as np
import argparse
import time
import os
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

try:
    import pyaudio
    import threading
    import queue
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("PyAudio not available. Microphone input disabled.")

import sys
import os
# Add parent directory to path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model'))

from models import get_model


class DigitClassifier:
    """
    Fast digit classifier for inference.
    
    Optimized for low latency with:
    - Efficient MFCC extraction
    - Model optimization (torch.jit)
    - Minimal preprocessing overhead
    """
    
    def __init__(
        self, 
        model_path: str,
        device: str = "auto",
        optimize_model: bool = True
    ):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            optimize_model: Whether to optimize model with torch.jit
        """
        # Device setup
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model on {self.device}")
        
        # Load model
        self.model = get_model()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Optimize model for inference
        if optimize_model:
            try:
                # Create dummy input for tracing
                dummy_input = torch.randn(1, 13, 32).to(self.device)
                self.model = torch.jit.trace(self.model, dummy_input)
                print("Model optimized with torch.jit")
            except Exception as e:
                print(f"Could not optimize model: {e}")
        
        # MFCC parameters (must match training)
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        self.max_length = 32
        self.sample_rate = 8000
        
        # Class names
        self.digit_names = [str(i) for i in range(10)]
        
        print("Digit classifier ready!")
    
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """
        Preprocess audio to MFCC features.
        
        Args:
            audio: Raw audio waveform
            sr: Sample rate of audio
            
        Returns:
            mfcc_tensor: Preprocessed MFCC features (1, 13, 32)
        """
        # Resample if necessary
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Ensure minimum length
        if len(audio) < self.hop_length:
            audio = np.pad(audio, (0, self.hop_length - len(audio)))
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True
        )
        
        # Normalize
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
        
        # Handle length normalization
        _, time_frames = mfcc.shape
        if time_frames < self.max_length:
            # Pad
            pad_width = ((0, 0), (0, self.max_length - time_frames))
            mfcc = np.pad(mfcc, pad_width, mode='constant', constant_values=0)
        elif time_frames > self.max_length:
            # Truncate from center
            start = (time_frames - self.max_length) // 2
            mfcc = mfcc[:, start:start + self.max_length]
        
        # Convert to tensor and add batch dimension
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).to(self.device)
        
        return mfcc_tensor
    
    def predict(self, audio: np.ndarray, sr: int) -> Tuple[int, float, np.ndarray]:
        """
        Predict digit from audio.
        
        Args:
            audio: Raw audio waveform
            sr: Sample rate
            
        Returns:
            predicted_digit: Predicted digit (0-9)
            confidence: Confidence score (0-1)
            probabilities: Full probability distribution
        """
        start_time = time.time()
        
        # Preprocess
        mfcc = self.preprocess_audio(audio, sr)
        
        # Inference
        with torch.no_grad():
            logits = self.model(mfcc)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_digit = predicted.item()
            confidence_score = confidence.item()
            prob_array = probabilities.cpu().numpy().flatten()
        
        inference_time = (time.time() - start_time) * 1000
        
        return predicted_digit, confidence_score, prob_array, inference_time
    
    def predict_file(self, audio_path: str, verbose: bool = True) -> Tuple[int, float]:
        """
        Predict digit from audio file.
        
        Args:
            audio_path: Path to audio file
            verbose: Whether to print results
            
        Returns:
            predicted_digit, confidence
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Predict
        digit, confidence, probabilities, inference_time = self.predict(audio, sr)
        
        if verbose:
            print(f"File: {os.path.basename(audio_path)}")
            print(f"Predicted digit: {digit}")
            print(f"Confidence: {confidence:.3f}")
            print(f"Inference time: {inference_time:.1f} ms")
            print(f"Probabilities: {probabilities}")
            print("-" * 40)
        
        return digit, confidence


class MicrophoneRecorder:
    """
    Clean microphone recording for digit classification.
    Supports both single recording and continuous listening modes.
    """
    
    def __init__(self, classifier: DigitClassifier):
        if not PYAUDIO_AVAILABLE:
            raise ImportError("PyAudio is required for microphone input")
        
        self.classifier = classifier
        self.sample_rate = 8000
        self.chunk_size = 1024
        self.record_duration = 2.0  # seconds per recording
        
        # Voice activity detection - more sensitive settings
        self.silence_threshold = 0.01  # Lower threshold for better detection
        self.min_speech_duration = 0.3  # Shorter minimum speech
        self.silence_duration = 1.0     # Longer silence buffer
        self.confidence_threshold = 0.85  # High confidence required
        
        print("Microphone ready for digit classification")
        
    def record_single(self):
        """Record a single digit (manual trigger)."""
        audio = pyaudio.PyAudio()
        
        print(f"Recording for {self.record_duration} seconds... Speak a digit!")
        
        # Record audio
        stream = audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        for _ in range(int(self.sample_rate / self.chunk_size * self.record_duration)):
            data = stream.read(self.chunk_size)
            frames.append(np.frombuffer(data, dtype=np.float32))
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Process audio
        audio_data = np.concatenate(frames)
        
        # Check if audio has enough energy (not just silence)
        audio_energy = np.sqrt(np.mean(audio_data**2))
        if audio_energy < self.silence_threshold:
            print("⚠️  Too quiet - please speak louder")
            return -1, 0.0
        
        print("Processing audio...")
        
        # Classify
        digit, confidence, probabilities, inference_time = self.classifier.predict(
            audio_data, self.sample_rate
        )
        
        # Apply confidence thresholding
        if confidence < self.confidence_threshold:
            print(f"❌ Low confidence ({confidence:.1%}) - please speak more clearly")
            print(f"   (Heard something like '{digit}' but not sure)")
            return -1, confidence
        
        # Show results for high confidence predictions
        print(f"🎯 Predicted digit: {digit}")
        print(f"✅ Confidence: {confidence:.1%}")
        print(f"⚡ Inference time: {inference_time:.1f}ms")
        
        return digit, confidence
    
    def start_continuous_listening(self):
        """Start continuous listening with voice activity detection."""
        import threading
        import queue
        
        print("🎤 Starting improved microphone recognition...")
        print("📋 Tips for best results:")
        print("   • Speak digits (0-9) clearly and loudly")
        print("   • Pause 1 second between digits")
        print("   • Avoid background noise")
        print(f"   • Only predictions ≥{self.confidence_threshold:.0%} confidence will be shown")
        print("📱 Press Ctrl+C to stop")
        print("-" * 50)
        
        # Audio queue for thread communication
        audio_queue = queue.Queue()
        is_listening = True
        
        def audio_callback(in_data, frame_count, time_info, status):
            if is_listening:
                audio_chunk = np.frombuffer(in_data, dtype=np.float32)
                audio_queue.put(audio_chunk)
            return (in_data, pyaudio.paContinue)
        
        # Start audio stream
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=audio_callback
        )
        
        stream.start_stream()
        
        # Process audio continuously
        current_recording = []
        silence_counter = 0
        speech_detected = False
        
        try:
            while is_listening:
                try:
                    audio_chunk = audio_queue.get(timeout=0.1)
                    
                    # Voice activity detection
                    volume = np.sqrt(np.mean(audio_chunk ** 2))
                    
                    if volume > self.silence_threshold:
                        # Speech detected
                        current_recording.extend(audio_chunk)
                        silence_counter = 0
                        speech_detected = True
                    else:
                        # Silence detected
                        if speech_detected:
                            silence_counter += 1
                            silence_duration = silence_counter * self.chunk_size / self.sample_rate
                            
                            # Process recording after sufficient silence
                            if silence_duration >= self.silence_duration and len(current_recording) > 0:
                                min_samples = int(self.min_speech_duration * self.sample_rate)
                                
                                if len(current_recording) >= min_samples:
                                    audio_data = np.array(current_recording, dtype=np.float32)
                                    
                                    # Classify
                                    digit, confidence, _, inference_time = self.classifier.predict(
                                        audio_data, self.sample_rate
                                    )
                                    
                                    # Only show high confidence predictions
                                    if confidence >= self.confidence_threshold:
                                        print(f"🎯 Detected: {digit}")
                                        print(f"✅ Confidence: {confidence:.1%}")
                                    elif confidence > 0.5:
                                        print(f"❓ Unclear speech (heard: {digit}, confidence: {confidence:.1%})")
                                    else:
                                        print(f"🔇 [Speech too unclear - skipped]")
                                
                                # Reset
                                current_recording = []
                                silence_counter = 0
                                speech_detected = False
                
                except queue.Empty:
                    continue
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping continuous listening...")
        finally:
            is_listening = False
            stream.stop_stream()
            stream.close()
            audio.terminate()


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Digit classification inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--mode', type=str, default='file', choices=['file', 'mic'],
                        help='Inference mode: file or microphone')
    parser.add_argument('--audio_file', type=str, 
                        help='Audio file path (for file mode)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = DigitClassifier(
        model_path=args.model_path,
        device=args.device
    )
    
    if args.mode == 'file':
        if not args.audio_file:
            print("Error: --audio_file required for file mode")
            return
        
        if not os.path.exists(args.audio_file):
            print(f"Error: Audio file {args.audio_file} not found")
            return
        
        # File inference
        print("=== File Inference ===")
        digit, confidence = classifier.predict_file(args.audio_file)
        
    elif args.mode == 'mic':
        if not PYAUDIO_AVAILABLE:
            print("Error: PyAudio not available for microphone input")
            print("Install with: pip install pyaudio")
            return
        
        # Microphone inference
        print("=== Microphone Inference ===")
        recorder = MicrophoneRecorder(classifier)
        
        print("\nChoose microphone mode:")
        print("1. Single recording (manual trigger)")
        print("2. Continuous listening (voice detection)")
        
        while True:
            try:
                choice = input("\nEnter choice (1/2) or 'q' to quit: ").strip()
                
                if choice == 'q':
                    break
                elif choice == '1':
                    input("Press Enter to start recording...")
                    recorder.record_single()
                elif choice == '2':
                    recorder.start_continuous_listening()
                    break  # Exit after continuous session
                else:
                    print("Invalid choice. Enter 1, 2, or 'q'")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break


if __name__ == "__main__":
    main()
