import random
import argparse

class IceCreamGenerator:
    def __init__(self, order=2):
        """
        Initialize the generator with the desired Markov chain order
        order: how many previous characters to consider when generating the next one
        """
        self.order = order
        self.transitions = {}
        
    def train(self, flavors):
        """
        Train the generator on a list of ice cream flavor names
        flavors: list of strings containing ice cream flavor names
        """
        # Reset transitions dictionary
        self.transitions = {}
        
        # Process each flavor
        for flavor in flavors:
            # Add start and end markers
            padded = "^" * self.order + flavor + "$"
            
            # Build transitions dictionary
            for i in range(len(padded) - self.order):
                # Get the current state (previous characters)
                current = padded[i:i + self.order]
                # Get the next character
                next_char = padded[i + self.order]
                
                # Add to transitions dictionary
                if current not in self.transitions:
                    self.transitions[current] = {}
                if next_char not in self.transitions[current]:
                    self.transitions[current][next_char] = 0
                self.transitions[current][next_char] += 1
    
    def generate(self, min_length=3, max_length=20, temperature=1.0, attempts=50):
        """
        Generate a new ice cream flavor name
        min_length: minimum characters in generated name
        max_length: maximum characters in generated name
        temperature: controls randomness (lower = more conservative)
        attempts: number of tries to generate a valid name
        """
        for _ in range(attempts):
            # Start with beginning marker
            result = "^" * self.order
            
            # Generate characters until we hit an end marker or max length
            while len(result) - self.order < max_length:
                current = result[-self.order:]
                
                # If we haven't seen this state before, start over
                if current not in self.transitions:
                    break
                
                # Apply temperature to transition probabilities
                choices = []
                weights = []
                for next_char, count in self.transitions[current].items():
                    choices.append(next_char)
                    weights.append(count ** (1.0 / temperature))
                
                # Choose next character based on weighted probabilities
                total = sum(weights)
                weights = [w/total for w in weights]
                next_char = random.choices(choices, weights=weights)[0]
                
                # If we hit the end marker and we're past minimum length, we're done
                if next_char == "$" and len(result) - self.order >= min_length:
                    return result[self.order:]
                elif next_char != "$":
                    result += next_char
                    
        # If we failed to generate a valid name, try again with shorter length
        if max_length > min_length:
            return self.generate(min_length, max_length - 1, temperature)
        return None

def generate_flavors(num_flavors=5, order=2, min_length=3, max_length=20, temperature=1.0):
    """Generate multiple ice cream flavors"""
    with open('icecreams.txt', 'r') as f:
        flavors = [line.strip() for line in f if line.strip()]
    
    # Create and train generator
    generator = IceCreamGenerator(order=order)
    generator.train(flavors)
    
    # Generate new flavors
    new_flavors = []
    for _ in range(num_flavors):
        flavor = generator.generate(min_length=min_length, max_length=max_length, temperature=temperature)
        if flavor:
            new_flavors.append(flavor)
    
    return new_flavors

def main():
    parser = argparse.ArgumentParser(description='Generate ice cream flavors using a Markov chain')
    
    # Model parameters
    parser.add_argument('--order', type=int, default=2,
                      help='Order of the Markov chain (default: 2)')
    
    # Generation parameters
    parser.add_argument('--num_flavors', type=int, default=5,
                      help='Number of flavors to generate (default: 5)')
    parser.add_argument('--min_length', type=int, default=3,
                      help='Minimum length of generated names (default: 3)')
    parser.add_argument('--max_length', type=int, default=20,
                      help='Maximum length of generated names (default: 20)')
    parser.add_argument('--temperature', type=float, default=1.0,
                      help='Sampling temperature - lower is more conservative (default: 1.0)')
    
    # Input/Output
    parser.add_argument('--input_file', type=str, default='icecreams.txt',
                      help='Input text file (default: icecreams.txt)')
    
    args = parser.parse_args()
    
    # Load data
    try:
        with open(args.input_file, 'r') as f:
            text = f.read().strip()
        flavors = [line.strip() for line in text.split('\n') if line.strip()]
    except FileNotFoundError:
        print(f"Error: Could not find input file '{args.input_file}'")
        return
    
    # Create and train generator
    generator = IceCreamGenerator(order=args.order)
    generator.train(flavors)
    
    print(f"\nGenerating {args.num_flavors} flavors with parameters:")
    print(f"Order: {args.order}")
    print(f"Temperature: {args.temperature}")
    print(f"Length range: {args.min_length}-{args.max_length}")
    print("\nGenerated flavors:")
    
    # Generate and print flavors
    for i in range(args.num_flavors):
        flavor = generator.generate(
            min_length=args.min_length,
            max_length=args.max_length,
            temperature=args.temperature
        )
        if flavor:
            print(f"{i+1}. {flavor}")

if __name__ == '__main__':
    main()
