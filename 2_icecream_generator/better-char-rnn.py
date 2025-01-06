import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
import random

class TextDataset(Dataset):
    def __init__(self, text, sequence_length):
        self.sequence_length = sequence_length
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Convert text to indices
        self.encoded = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)
        
        # Create sequences
        self.n_sequences = len(text) - sequence_length
        
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        # Get sequence and target
        sequence = self.encoded[idx:idx + self.sequence_length]
        target = self.encoded[idx + 1:idx + self.sequence_length + 1]
        return sequence, target

class ImprovedCharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, 
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Initialize hidden state if needed
        if hidden is None:
            hidden = self.init_hidden(batch_size)
            
        # Embed input
        embedded = self.embedding(x)
        
        # LSTM forward pass
        output, hidden = self.lstm(embedded, hidden)
        
        # Apply dropout and final layer
        output = self.dropout(output)
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                 weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        return hidden

    def sample(self, dataset, seed_char='\n', length=100, temperature=1.0):
        self.eval()
        with torch.no_grad():
            # Initialize with seed character
            char_indices = [dataset.char_to_idx[seed_char]]
            hidden = None
            
            # Generate characters
            for _ in range(length):
                # Prepare input
                x = torch.tensor([char_indices[-dataset.sequence_length:]], dtype=torch.long)
                if len(x[0]) < dataset.sequence_length:
                    padding = torch.tensor([dataset.char_to_idx[seed_char]] * 
                                         (dataset.sequence_length - len(x[0])))
                    x = torch.cat([padding.unsqueeze(0), x], dim=1)
                
                # Forward pass
                output, hidden = self(x, hidden)
                output = output[0, -1] / temperature
                probs = torch.softmax(output, dim=-1)
                
                # Sample next character
                next_char_idx = torch.multinomial(probs, 1).item()
                char_indices.append(next_char_idx)
            
            # Convert indices back to characters
            return ''.join(dataset.idx_to_char[idx] for idx in char_indices[1:])

def train(model, dataset, args):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    from time import time
    start_time = time()
    last_print = start_time
    
    print("Training (printing progress every 10 seconds)...")
    
    for epoch in range(args.iterations):
        model.train()
        total_loss = 0
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output.view(-1, dataset.vocab_size), y.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress every 10 seconds
        current_time = time()
        if current_time - last_print >= 10:
            avg_loss = total_loss / len(dataloader)
            elapsed = current_time - start_time
            remaining = (elapsed / (epoch + 1)) * (args.iterations - epoch - 1)
            print(f'Iteration {epoch+1}/{args.iterations}, Loss: {avg_loss:.4f}')
            print(f'Time elapsed: {elapsed:.0f}s, Estimated remaining: {remaining:.0f}s')
            print('Sample:')
            print(model.sample(dataset, length=50, temperature=args.temperature))
            print()
            last_print = current_time
            
        # Also print at regular intervals
        elif (epoch + 1) % (args.iterations // 5) == 0:
            avg_loss = total_loss / len(dataloader)
            print(f'Iteration {epoch+1}, Loss: {avg_loss:.4f}')
            print('Sample:')
            print(model.sample(dataset, length=50, temperature=args.temperature))
            print()

def main():
    parser = argparse.ArgumentParser(description='Train an improved character-level RNN')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Size of hidden layer (default: 256)')
    parser.add_argument('--sequence_length', type=int, default=25,
                        help='Length of training sequences (default: 25)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    
    # Training parameters
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of training iterations (default: 1000)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (default: 0.7)')
    
    # Input/Output
    parser.add_argument('--input_file', type=str, default='icecreams.txt',
                        help='Input text file (default: icecreams.txt)')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    try:
        with open(args.input_file, 'r') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find input file '{args.input_file}'")
        return
    
    # Create dataset
    dataset = TextDataset(text, args.sequence_length)
    
    # Create model
    model = ImprovedCharRNN(
        vocab_size=dataset.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=2  # Using 2 LSTM layers
    )
    
    print("\nTraining with parameters:")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Temperature: {args.temperature}")
    print("\nStarting training...\n")
    
    # Train the model
    train(model, dataset, args)

if __name__ == '__main__':
    main()