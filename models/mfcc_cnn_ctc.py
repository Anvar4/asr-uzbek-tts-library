"""
Model 1: MFCC + CNN + CTC
Bu model klassik feature extraction (MFCC) va zamonaviy deep learning (CNN+CTC) ni birlashtiradi.
"""

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MFCCFeatureExtractor:
    """MFCC feature extraction"""
    
    def __init__(
        self, 
        sample_rate=16000,
        n_mfcc=40,
        n_fft=512,
        hop_length=160,
        n_mels=80
    ):
        self.sample_rate = sample_rate
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'n_mels': n_mels,
                'hop_length': hop_length
            }
        )
    
    def extract(self, waveform):
        """
        Args:
            waveform: (batch, time) or (time,)
        Returns:
            mfcc: (batch, n_mfcc, time) or (n_mfcc, time)
        """
        return self.mfcc_transform(waveform)


class MFCC_CNN_CTC(nn.Module):
    """
    Arxitektura:
    MFCC → CNN Blocks → BiLSTM → Fully Connected → CTC Loss
    """
    
    def __init__(
        self,
        input_dim=40,      # MFCC features
        hidden_dim=256,
        num_layers=3,
        num_classes=33,    # O'zbek alifbosi + blank + space
        dropout=0.3
    ):
        super(MFCC_CNN_CTC, self).__init__()
        
        # CNN Blocks for feature extraction
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )
        
        # Calculate CNN output dimension
        # input_dim=40 → 40/2/2/2 = 5
        cnn_output_dim = (input_dim // 8) * 128
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
        # Log Softmax for CTC
        self.log_softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_mfcc, time)
        Returns:
            log_probs: (time, batch, num_classes)
        """
        # Add channel dimension: (batch, 1, n_mfcc, time)
        x = x.unsqueeze(1)
        
        # CNN: (batch, 128, n_mfcc/8, time/8)
        x = self.cnn(x)
        
        # Reshape for LSTM: (batch, time/8, features)
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch, time, channels * freq)
        
        # LSTM: (batch, time, hidden*2)
        x, _ = self.lstm(x)
        
        # Fully connected: (batch, time, num_classes)
        x = self.fc(x)
        
        # Log softmax: (batch, time, num_classes)
        x = self.log_softmax(x)
        
        # CTC expects (time, batch, num_classes)
        x = x.permute(1, 0, 2)
        
        return x


class CTCLoss(nn.Module):
    """CTC Loss wrapper"""
    
    def __init__(self, blank=0):
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True)
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Args:
            log_probs: (T, N, C) - model output
            targets: (sum(target_lengths)) - ground truth
            input_lengths: (N,) - length of each sequence in batch
            target_lengths: (N,) - length of each target
        """
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Training loop"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        mfcc = batch['mfcc'].to(device)           # (batch, n_mfcc, time)
        targets = batch['targets'].to(device)      # (sum(target_lengths))
        input_lengths = batch['input_lengths']     # (batch,)
        target_lengths = batch['target_lengths']   # (batch,)
        
        optimizer.zero_grad()
        
        # Forward pass
        log_probs = model(mfcc)  # (time, batch, num_classes)
        
        # Calculate loss
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def decode_predictions(log_probs, vocab):
    """
    Simple greedy CTC decoder
    
    Args:
        log_probs: (T, N, C)
        vocab: list of characters
    Returns:
        decoded texts
    """
    # Get best path
    preds = torch.argmax(log_probs, dim=2)  # (T, N)
    preds = preds.permute(1, 0)  # (N, T)
    
    decoded_texts = []
    for pred in preds:
        # Remove blanks and duplicates
        decoded = []
        prev = None
        for p in pred:
            if p != 0 and p != prev:  # 0 is blank
                decoded.append(vocab[p.item()])
            prev = p
        decoded_texts.append(''.join(decoded))
    
    return decoded_texts


# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # O'zbek alifbosi
    uzbek_alphabet = [
        '<blank>',  # 0 - CTC blank
        ' ',         # 1 - space
        'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z',
        'o\'', 'g\'', 'sh', 'ch', 'ng', 'ye', 'yu', 'ya'
    ]
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MFCC_CNN_CTC(
        input_dim=40,
        hidden_dim=256,
        num_layers=3,
        num_classes=len(uzbek_alphabet)
    ).to(device)
    
    # Loss and optimizer
    criterion = CTCLoss(blank=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Example forward pass
    batch_size = 4
    n_mfcc = 40
    time_steps = 200
    
    dummy_input = torch.randn(batch_size, n_mfcc, time_steps).to(device)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # (time, batch, num_classes)
