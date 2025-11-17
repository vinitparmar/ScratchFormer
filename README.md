# 🧩 ScratchFormer

**ScratchFormer** is a Transformer neural network built **entirely from scratch** using **NumPy**.  
This project helps you deeply understand how modern architectures like GPT and BERT actually work — by implementing every part manually, from attention to embeddings.

---

## 🚀 Features
- ✅ Pure **NumPy** implementation (no deep learning frameworks)
- 🧠 Full **Encoder–Decoder Transformer** architecture
- 🔍 Implements:
  - Scaled Dot-Product & Multi-Head Attention  
  - Positional Encoding (sinusoidal)  
  - Feed Forward Networks (GELU activation)  
  - Layer Normalization & Residual Connections
- 🧩 Modular and easy to extend
- 💡 Educational, readable, and well-documented
- 🧪 Includes toy training example (copy / translation task)

---
HASE 1 — Foundations

📅 Time: 1–2 days

1. Set up repository

Folder structure (scratchformer/, examples/, notebooks/)

Add README + requirements

Create empty module files

2. Write utility math functions

softmax

stable softmax (numerically safe)

create masks (padding + causal)

matrix operations (optional helpers)

📌 Goal: be comfortable with matrix shapes (batch, seq, dim).

PHASE 2 — Core Components (Building Blocks)

📅 Time: 3–5 days

We code every block manually.

3. Token Embedding

lookup matrix vocab_size × d_model

convert token IDs → vectors

4. Positional Encoding

sinusoidal positional encoding (NumPy)

add to embeddings

5. Scaled Dot-Product Attention

compute Q, K, V

formula:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
$$

test with small input

6. Multi-Head Attention

linear projection into heads

split → attention → concat

output projection

ensure shapes match exactly

7. Feed Forward Network

Dense → GELU → Dense

per-position (works on each token independently)

8. Layer Normalization

implement from scratch:

mean

variance

normalize

gamma, beta parameters

📌 Goal: Each block should be testable alone with a small script.

PHASE 3 — Encoder-Decoder Architecture

📅 Time: 3–5 days

9. Encoder Layer

multi-head self-attention

residual + layernorm

feed-forward

residual + layernorm

test with random tokens

10. Decoder Layer

masked self-attention

encoder–decoder attention

feed-forward

residuals + norms

11. Encoder Stack

stack N layers in a loop

12. Decoder Stack

stack N layers in a loop

📌 Goal: Build full working encoder & decoder.

PHASE 4 — Full Transformer Model

📅 Time: 3–4 days

13. Combine encoder + decoder

input embeddings

positional encodings

encoder output → decoder input

final linear layer projecting to vocab size

14. Forward pass

accept:

src_tokens

tgt_tokens

masks

output logits

15. Greedy decoding

autoregressive decoding

feed previous tokens into decoder

generate sequences

📌 Goal: Model can run a forward pass and generate output.

PHASE 5 — Training (Toy Examples)

📅 Time: 3–6 days

16. Build simple cross-entropy loss

compute average loss ignoring padding tokens

17. Create toy dataset

Examples:

Copy task (Y = X)

Reverse task (Y = reversed(X))

Shift-by-one task

Tiny translation mapping

18. Training loop

forward pass

compute loss

backprop through NumPy (optional)

OR partially use PyTorch autograd

update weights manually (SGD or Adam)

📌 Goal: Loss should go down after 5–20 epochs.

PHASE 6 — Enhancements (Optional but Powerful)

📅 Time: 1–2 weeks

19. Port model to PyTorch

same architecture, easier training

GPUs + autograd

20. Visualize Attention

plot attention matrices

use matplotlib or seaborn

21. Add support for larger configs

more layers

more attention heads

22. Train on a real dataset

small translation dataset (IWSLT)

character-level modeling

mini GPT

PHASE 7 — Release & Document

📅 Time: 1–2 days

23. Clean codebase

comments

modular structure

remove unused code

24. Final README updates

diagrams (like the one generated)

architecture explanation

code examples

formulas

---

## 🏗️ Project Structure

  ScratchFormer/
  │
  ├── README.md
  ├── requirements.txt
  │
  ├── scratchformer/
  │ ├── init.py
  │ └── transformer_from_scratch_numpy.py # Main transformer implementation
  │
  ├── examples/
  │ └── copy_task_demo.py # Simple demo/training example
  │
  └── notebooks/
  └── transformer_from_scratch.ipynb # (Optional) Jupyter notebook for explanation


## ⚙️ Installation & Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/<your-username>/ScratchFormer.git
   cd ScratchFormer

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Run the demo**
   ```bash
   python examples/copy_task_demo.py

---
## Block Diagram

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/574f9775-ec8f-4376-a7f2-862f22190a54" />

🧱 1. Embedding & Positional Encoding

Goal: Convert token IDs into continuous vector representations.

We'll code:

Token Embedding

Sinusoidal Positional Encoding

➡️ Output: tensor of shape (batch_size, seq_len, d_model)

⚡ 2. Scaled Dot-Product Attention

Goal: Compute attention weights between tokens.

We'll code:

Queries (Q), Keys (K), Values (V)

Attention formula

### 🧠 Scaled Dot-Product Attention

The core operation behind the Transformer is the **Scaled Dot-Product Attention**, defined as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
$$

Where:
- \( Q \) = Query matrix  
- \( K \) = Key matrix  
- \( V \) = Value matrix  
- \( d_k \) = dimensionality of the key vectors


➡️ Output: context vectors (weighted representations)

🧩 3. Multi-Head Attention

Goal: Run multiple attention heads in parallel.

We'll code:

Linear projections for multiple heads

Head splitting and concatenation

Output projection layer

➡️ Output: richer contextual embeddings

🔁 4. Feed Forward Network (FFN)

Goal: Add non-linear transformations per token.

We'll code:

Linear → GELU → Linear

Dropout (optional)

➡️ Output: transformed representation per position

🧠 5. Layer Normalization + Residual Connections

Goal: Stabilize and accelerate training.

We'll code:

LayerNorm(x + Sublayer(x))

Residual skips between attention and feed-forward layers

🧰 6. Encoder Layer

Goal: Stack multiple layers of attention + FFN.

We'll code:

Self-Attention + FFN + Norm + Residual

N identical layers in sequence

💬 7. Decoder Layer

Goal: Generate sequences autoregressively.

We'll code:

Masked Multi-Head Self-Attention

Encoder–Decoder Attention

Feed Forward Network

Norm + Residual

🧩 8. Transformer (Full Model)

Goal: Combine encoder + decoder into one model.

We'll code:

Encoder → Decoder → Linear Projection → Softmax

➡️ Output: logits over target vocabulary.

🔬 9. Training Loop (Toy Task)

Goal: See your model learn something.

We'll code:

Forward pass

Cross-entropy loss

Optimization step

Evaluate toy copy/translate task


---

🧠 Concepts You’ll Learn

    Linear Algebra in neural networks (matrix operations)
    
    Self-Attention and Multi-Head Attention
    
    Positional Encoding and Sequence Order
    
    Encoder–Decoder architecture
    
    Layer Normalization and Residual Connections
    
    How Transformers learn sequence relationships

---

📘 Educational Goals

    ScratchFormer was built to:
    
    Teach the core principles of Transformers
    
    Provide a readable, minimal implementation for learning
    
    Serve as a foundation for experiments or PyTorch conversion
    
    Help developers, students, and AI enthusiasts understand every line of a Transformer

---

📚 References

    Attention Is All You Need (Vaswani et al., 2017)
     url-> https://arxiv.org/abs/1706.03762
    
    The Illustrated Transformer — Jay Alammar
     url-> https://jalammar.github.io/illustrated-transformer/
    
    3Blue1Brown — Linear Algebra Series**
     url-> https://www.3blue1brown.com/topics/linear-algebra
      


---

🧰 Requirements

    Python 3.9+
    NumPy 1.26+

---

🪄 Future Enhancements

    PyTorch version for training with autograd
    
    Visualization of attention maps
    
    GPT-style text generation
    
    Colab notebook for interactive learning

---

📄 License

    MIT License © 2025 Vinit Parmar
