## EXERCISE: Three Approaches to Ice Cream Flavor Generation

In this exercise, you'll explore three different implementations of text generation, each with increasing sophistication:

1. Markov Chain (`markov-ice-cream.py`)
2. Basic RNN (`minimal-char-rnn.py`)
3. LSTM Network (`better-char-rnn.py`)

## PART 1: Understanding the Approaches

### Markov Chain
The simplest approach. A Markov Chain remembers sequences of characters and their probabilities.

How it works:
```
Given: "CHOCOLATE"
With order=2, it learns:
CH -> O (100%)
HO -> C (100%)
OC -> O (100%)
CO -> L (100%)
OL -> A (100%)
LA -> T (100%)
AT -> E (100%)
```

Run with:
```bash
python markov-ice-cream.py --order 3 --temperature 0.7
```

Expected results:
```
Chocolate
Vanilla Bean
Strawberry
```

### Basic RNN (Recurrent Neural Network)
A simple neural network that processes sequences by maintaining a hidden state. It struggles because simple RNNs have trouble learning long-term patterns.

How it works:
- Takes one character at a time
- Updates a hidden state
- Predicts next character
- Often gets stuck due to "vanishing gradients"

Run with:
```bash
python minimal-char-rnn.py --hidden_size 150 --sequence_length 12
```

Expected results:
```
eeeee kr
seeeesienoon
heaeerSeea
```

### LSTM Network (Long Short-Term Memory)
An improved RNN that solves the vanishing gradient problem using specialized memory cells.

How it works:
- Has additional "gates" that control information flow:
  * Forget gate: What to remove from memory
  * Input gate: What new info to store
  * Output gate: What to use for predictions
- Can learn longer patterns
- Better at preserving important information

Run with:
```bash
python better-char-rnn.py --hidden_size 256 --sequence_length 25
```

Expected results:
```
Chocolate Mint
Strawberry Swirl
Vanilla Bean Cookie
```

## PART 2: Understanding What You See

When you run each script, here's what's happening:

### Markov Chain Output:
```bash
$ python markov-ice-cream.py
Order: 3
Temperature: 0.7
Generated flavors:
1. Chocolate Swirl
2. Strawberry
3. Vanilla Bean
```
- Immediately starts generating names
- Results are clear and readable
- Often reuses exact phrases from training data

### Basic RNN Output:
```bash
$ python minimal-char-rnn.py
Iteration 100, Loss: 85.4332
Sample: eeeee kr seeeesienoon
Iteration 200, Loss: 65.2211
Sample: heaeerSeea neanrsel
```
- Shows loss decreasing as it trains
- Samples start random, slowly develop patterns
- Often gets stuck repeating characters
- Struggles with word structure

### LSTM Output:
```bash
$ python better-char-rnn.py
Iteration 100, Loss: 2.4521
Sample: Choclate Swirl
Iteration 200, Loss: 1.8765
Sample: Strawberry Cream
```
- Loss decreases more consistently
- Learns word patterns faster
- Maintains coherent word structure
- Can create novel combinations



## METRICS


## Iterations
An iteration is one complete pass through a batch of training data. When you see:
```
Iteration 100/500, Loss: 2.4521
```
This means we're on the 100th training cycle out of 500 planned cycles.

Think of it like practice rounds:
- Each iteration, the model sees a batch of ice cream names
- It tries to predict each next character
- It updates its understanding based on how wrong it was
- More iterations = more practice

## Loss

Loss measures how badly the model is doing at its predictions. When you see:
```
Loss: 2.4521 -> Loss: 1.8765 -> Loss: 1.2234
```
- Lower numbers are better
- It starts high (lots of mistakes)
- It should decrease as training progresses
- For character prediction, anything under 2.0 means it's learning patterns

For ice cream names, the loss represents:
- How surprised the model is by each next character
- Higher loss = "I didn't expect that character at all!"
- Lower loss = "I was pretty sure that 'a' would follow after 'Vanill'"

For example, if you see:
```
Iteration 50/500, Loss: 3.2456
Sample: Xkqpzm

Iteration 150/500, Loss: 2.1234
Sample: Vanila

Iteration 250/500, Loss: 1.7654
Sample: Vanilla Bean
```
You can see the relationship between decreasing loss and improving output quality. The model is getting better at predicting what characters should come next in ice cream names.



PART 3: Experimentation Tasks
---------------------------

1. Temperature Effects
Try each implementation with different temperatures:
```bash
--temperature 0.2  # Very conservative
--temperature 0.7  # Balanced
--temperature 1.5  # Very creative
```

2. Pattern Length
Experiment with how much context each model uses:
```bash
# Markov
--order 2  # Short patterns
--order 4  # Longer patterns

# RNNs
--sequence_length 10  # Short sequences
--sequence_length 30  # Longer sequences
```

3. Model Size (LSTM only)
Try different network sizes:
```bash
--hidden_size 128  # Smaller network
--hidden_size 512  # Larger network
```

Questions to Answer:
1. How does output quality change as you increase pattern length/sequence length?
2. Which model is most sensitive to temperature changes?
3. What happens when you make the LSTM too small or too large?
4. Which model creates the most novel (but still plausible) combinations?

PART 4: Understanding the Results
--------------------------------

Why such different results?

Markov Chain:
- Pros: Exact copies of local patterns
- Cons: No understanding of patterns beyond order length
- Best for: Maintaining exact local structure

Basic RNN:
- Pros: Attempts to learn patterns
- Cons: Struggles with long-term dependencies
- Best for: Understanding basic concepts but not practical for this task

LSTM:
- Pros: Can learn and maintain long patterns
- Cons: Needs more data/training than Markov Chain
- Best for: Creative combinations while maintaining structure

This demonstrates how different architectures handle the same task, and why modern text generation usually uses LSTM or Transformer architectures rather than basic RNNs.

