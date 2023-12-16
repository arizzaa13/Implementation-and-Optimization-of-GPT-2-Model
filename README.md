# Implementation-and-Optimization-of-GPT-2-Model
This repository contains Implementation and Optimization of GPT-2 Model through transformer architecture and modifying its structure for improved performance further it includes developing an efficient training loop and implementation of distributed training applicable across multiple GPUs.

# TASK-1
# GPT-2 Model & Checkpoints 

At first we start by implementing the GPT-2 model. We will create a basic version of the GPT-2-small model with 125 million parameters.This implementation is a simplified version for demonstration purposes, and we may need to adapt it based on our specific requirements.

# TASK-2
# Transformer Architectural Changes 

Adding alterations to our original GPT-2 model architecture to experiment and assess the potential of improvements.

- **Rotary Positional Embedding:** Replacing the original positional embeddings in the GPT-2 model with Rotary embeddings.
- **Group Query Attention:** Equip model with the Group Query Attention mechanism 
- **Sliding Window Attention:** Imbibe the Sliding Window Attention mechanism in model and observe its effects on model performance.

## Task 3: Training Loop Implementation (40 Points)

---

Finally, create a training loop considering these following requirements:

1. **Single GPU Training Loop:** base implementation should be equipped to train model on a single GPU setup.
2. **Distributed Data Parallel (DDP):** Extend single GPU training loop to support training across multiple GPUs using DDP. 
3. **Fully Sharded Data Parallel (FSDP):** Implement FSDP as a part of your training loop to shard the model parameters, gradients, and optimizer state.
