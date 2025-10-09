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

The Illustrated Transformer — Jay Alammar

3Blue1Brown — Linear Algebra Series


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
