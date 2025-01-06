import numpy as np
import argparse

def sigmoid(x):
    """Compute sigmoid activation"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Compute derivative of sigmoid"""
    return x * (1 - x)

class CharacterRNN:
    def __init__(self, hidden_size=100, sequence_length=25, learning_rate=0.1):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        
        # Model parameters will be initialized when we see the vocabulary
        self.W_xh = None  # input to hidden
        self.W_hh = None  # hidden to hidden
        self.W_hy = None  # hidden to output
        self.b_h = None   # hidden bias
        self.b_y = None   # output bias
        
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
    def init_parameters(self, vocab_size):
        """Initialize network parameters"""
        self.vocab_size = vocab_size
        
        # Initialize weights with small random values
        self.W_xh = np.random.randn(self.hidden_size, vocab_size) * 0.01
        self.W_hh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.W_hy = np.random.randn(vocab_size, self.hidden_size) * 0.01
        
        # Initialize biases to zero
        self.b_h = np.zeros((self.hidden_size, 1))
        self.b_y = np.zeros((vocab_size, 1))
    
    def prepare_data(self, text):
        """Create character mappings from input text"""
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        if self.vocab_size == 0:
            self.init_parameters(len(chars))
            
        return chars
    
    def encode_text(self, text):
        """Convert text to one-hot encoded vectors"""
        encoded = []
        for char in text:
            vector = np.zeros((self.vocab_size, 1))
            vector[self.char_to_idx[char]] = 1
            encoded.append(vector)
        return encoded
    
    def forward(self, inputs, h_prev):
        """
        Forward pass through the network
        inputs: list of one-hot vectors for input sequence
        h_prev: initial hidden state
        """
        h, y = {}, {}
        h[-1] = np.copy(h_prev)
        
        # Forward pass
        for t in range(len(inputs)):
            # Hidden state
            h[t] = sigmoid(np.dot(self.W_xh, inputs[t]) + 
                         np.dot(self.W_hh, h[t-1]) + self.b_h)
            
            # Output probabilities
            y[t] = np.dot(self.W_hy, h[t]) + self.b_y
            y[t] = np.exp(y[t]) / np.sum(np.exp(y[t]))  # softmax
            
        return h, y
    
    def sample(self, h, seed_char, n_chars, temperature=1.0):
        """Generate text starting with seed_char"""
        x = np.zeros((self.vocab_size, 1))
        x[self.char_to_idx[seed_char]] = 1
        generated = seed_char
        
        for _ in range(n_chars):
            # Forward pass
            h = sigmoid(np.dot(self.W_xh, x) + np.dot(self.W_hh, h) + self.b_h)
            y = np.dot(self.W_hy, h) + self.b_y
            # Apply temperature scaling
            y = y / temperature
            p = np.exp(y) / np.sum(np.exp(y))
            
            # Sample from probability distribution
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())
            generated += self.idx_to_char[idx]
            
            # Prepare next input
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            
        return generated
    
    def train(self, text, n_iterations=100, decay_learning_rate=False, print_every=100,
             sample_length=100, temperature=1.0):
        """Train the network"""
        chars = self.prepare_data(text)
        data = self.encode_text(text)
        n = 0
        smooth_loss = -np.log(1.0/self.vocab_size) * self.sequence_length
        
        while n < n_iterations:
            # Update learning rate if decay is enabled
            if decay_learning_rate:
                self.learning_rate = self.initial_learning_rate * (0.9 ** (n // 100))
            
            # Get random starting position
            pos = np.random.randint(0, len(data) - self.sequence_length - 1)
            inputs = data[pos:pos + self.sequence_length]
            targets = data[pos + 1:pos + self.sequence_length + 1]
            
            h = np.zeros((self.hidden_size, 1))
            loss = 0
            
            # Forward pass
            h_states, y_preds = self.forward(inputs, h)
            
            # Backward pass (simplified)
            dW_xh = np.zeros_like(self.W_xh)
            dW_hh = np.zeros_like(self.W_hh)
            dW_hy = np.zeros_like(self.W_hy)
            db_h = np.zeros_like(self.b_h)
            db_y = np.zeros_like(self.b_y)
            dh_next = np.zeros_like(h_states[0])
            
            # For each time step
            for t in reversed(range(len(inputs))):
                # Gradient of output
                dy = np.copy(y_preds[t])
                dy[np.argmax(targets[t])] -= 1
                
                # Update gradients
                dW_hy += np.dot(dy, h_states[t].T)
                db_y += dy
                
                dh = np.dot(self.W_hy.T, dy) + dh_next
                dh_raw = sigmoid_derivative(h_states[t]) * dh
                
                db_h += dh_raw
                dW_xh += np.dot(dh_raw, inputs[t].T)
                dW_hh += np.dot(dh_raw, h_states[t-1].T)
                
                dh_next = np.dot(self.W_hh.T, dh_raw)
                
                loss -= np.log(y_preds[t][np.argmax(targets[t])])
            
            # Update weights using gradient descent
            for param, dparam in zip([self.W_xh, self.W_hh, self.W_hy, self.b_h, self.b_y],
                                   [dW_xh, dW_hh, dW_hy, db_h, db_y]):
                param -= self.learning_rate * dparam
            
            # Convert loss to float for smooth calculation
            loss_value = float(loss)
            smooth_loss = smooth_loss * 0.999 + loss_value * 0.001
            
            if n % print_every == 0:
                print(f'Iteration {n}, Loss: {float(smooth_loss):.4f}, Learning Rate: {float(self.learning_rate):.4f}')
                print(f'Sample (temp={temperature:.2f}):')
                print(self.sample(h, text[0], sample_length, temperature))
                print('\n')
            
            n += 1

def main():
    parser = argparse.ArgumentParser(description='Train a character-level RNN on text data')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=100,
                        help='Size of hidden layer (default: 100)')
    parser.add_argument('--sequence_length', type=int, default=25,
                        help='Length of training sequences (default: 25)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    
    # Training parameters
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of training iterations (default: 1000)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0)')
    parser.add_argument('--sample_length', type=int, default=100,
                        help='Length of sampled text during training (default: 100)')
    parser.add_argument('--print_every', type=int, default=100,
                        help='Print sample every N iterations (default: 100)')
    parser.add_argument('--decay_learning_rate', action='store_true',
                        help='Enable learning rate decay')
    
    # Input/Output
    parser.add_argument('--input_file', type=str, default='icecreams.txt',
                        help='Input text file (default: icecreams.txt)')
    
    args = parser.parse_args()
    
    # Load data
    try:
        with open(args.input_file, 'r') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find input file '{args.input_file}'")
        return
    
    # Create and train the RNN
    rnn = CharacterRNN(
        hidden_size=args.hidden_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate
    )
    
    print("Training with parameters:")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Temperature: {args.temperature}")
    print(f"Learning rate decay: {args.decay_learning_rate}")
    print("\nStarting training...\n")
    
    rnn.train(
        text,
        n_iterations=args.iterations,
        decay_learning_rate=args.decay_learning_rate,
        print_every=args.print_every,
        sample_length=args.sample_length,
        temperature=args.temperature
    )

if __name__ == '__main__':
    main()