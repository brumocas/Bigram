# Bigram Language Model

A minimal implementation of a bigram language model built from scratch using PyTorch. This is the simplest possible neural language model - a foundational example that demonstrates the basic concepts of language modeling before moving to more complex architectures like transformers.

## Overview

This implementation features:
- **Character-level tokenization** for text processing
- **Embedding-based bigram model** that predicts the next token given the current one
- **Simple architecture** with just an embedding lookup table
- **Training and text generation** capabilities

## Architecture

The model is extremely simple:
- **Token Embedding Table**: A lookup table that maps each character directly to logits for predicting the next character
- No attention mechanism, no transformer blocks, no positional encodings
- The model learns character-to-character transition probabilities

### Model Components

- `BigramLanguageModel`: A single embedding layer that serves as both the input representation and output prediction head

## How It Works

Despite the name "bigram" (which typically refers to pairs of tokens), this model uses a context window (`block_size`) but doesn't effectively utilize it since there's no mechanism to process sequences. Each token independently predicts the next token through the embedding lookup. The model learns which characters are likely to follow other characters based on the training data.

## Files

- `bigram.py`: Complete implementation including model definition, training loop, and text generation
- `input.txt`: Training dataset (text corpus)
- `bigram.pth`: Saved model weights (after training)

## Requirements

- Python 3.x
- PyTorch

## Usage

1. **Prepare your dataset**: Place your text corpus in `input.txt` (or download the TinyShakespeare dataset as mentioned in the code)

2. **Train the model**:
   ```bash
   python bigram.py
   ```

3. **Model Configuration**: The training script includes configurable hyperparameters:
   - `batch_size`: Number of sequences processed in parallel (default: 32)
   - `block_size`: Maximum context length (default: 8)
   - `learning_rate`: Learning rate for AdamW optimizer (default: 1e-2)
   - `max_iters`: Maximum training iterations (default: 3000)
   - `eval_interval`: How often to evaluate on validation set (default: 300)

4. **Output**: After training, the model will:
   - Print generated text to the console (500 tokens)
   - Save weights to `bigram.pth`

## Training Details

- Uses character-level tokenization (builds vocabulary from unique characters in the dataset)
- 90/10 train/validation split
- AdamW optimizer
- Evaluates loss on validation set every 300 iterations
- Automatically uses CUDA if available, otherwise falls back to CPU

## Model Size

The model has approximately `vocab_size²` parameters (one embedding per character, with vocab_size output logits for each).

## Educational Purpose

This is a minimal educational implementation that serves as a stepping stone to understanding more complex language models. It demonstrates:
- Basic language modeling concepts
- Character-level tokenization
- Training loops and loss computation
- Text generation through sampling

For better text generation quality, see the GPT implementation in the parent directory, which adds attention mechanisms and transformer architecture.