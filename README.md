# 🧠 LSTM From Scratch (Sequence Copy Task)

### Overview

This project implements a **Long Short-Term Memory (LSTM)** neural network **entirely from scratch using NumPy** — without relying on deep learning frameworks like PyTorch or TensorFlow.

The goal: **learn to copy input binary sequences** (e.g., `[1,1,0,1] → [1,1,0,1]`) by modeling sequence dependencies and memory transitions through manual implementation of LSTM gates and cell updates.

This project was built purely for **learning and demonstrating deep understanding** of recurrent neural networks and how LSTMs handle long-term dependencies.

---

### 🎯 Project Highlights

* 🧩 Implemented **Forget, Input, Cell, and Output Gates** manually using NumPy operations.
* ⚙️ Built custom **activation functions (`sigmoid`, `tanh`)** and their derivatives.
* 🔁 Designed **forward pass** logic to propagate information through time steps.
* 🧮 Trained the model on random binary-encoded integers to copy sequences.
* 🔍 Validated the LSTM’s internal states (hidden & cell states) and output behavior.
* 💡 Demonstrates **under-the-hood understanding of Backpropagation Through Time (BPTT)** and recurrent computation flow.

---

### 🧰 Tech Stack

* **Language:** Python
* **Libraries:** NumPy, Matplotlib (for visualization, optional)
* **Environment:** Jupyter Notebook / IPython

---

### 🚀 How It Works

1. **Dataset Generation:**
   Random integers are generated and converted into **fixed-width binary sequences** (e.g., `25 → [0,0,0,1,1,0,0,1]`).
   The model’s goal is to reproduce the same sequence as output.

2. **Model Architecture:**
   The `Copy_Lstm` class implements:

   * Forget gate: decides what past information to discard.
   * Input gate: decides what new information to store.
   * Cell state update: combines memory updates.
   * Output gate: determines what to output at each time step.
   * Final linear layer for prediction.

3. **Training Loop (optional):**
   Model processes multiple examples to minimize reconstruction error.
   Can be extended to support gradient updates (currently conceptual focus).

4. **Evaluation:**
   Test sequences are passed to verify whether the model correctly reproduces input patterns.

---

### 🧩 Sample Input / Output

| Input Sequence | Predicted Output |
| -------------- | ---------------- |
| [1, 1, 0, 1]   | [1, 1, 0, 1]     |
| [0, 1, 1, 0]   | [0, 1, 1, 0]     |

*(Results approximate depending on training and parameter initialization)*

---

### 📚 Learning Outcomes

* Deep understanding of **how LSTMs store and update memory** across time steps.
* Practical experience with **matrix operations and activation gradients**.
* Appreciation of how frameworks like TensorFlow simplify these low-level details.
* Strengthened intuition for **sequence modeling and recurrent computation.**

---

### 🧭 Future Improvements

* Implement **Backpropagation Through Time (BPTT)** to train weights end-to-end.
* Extend to **many-to-one** or **many-to-many** sequence tasks (e.g., text prediction).
* Compare performance with **PyTorch’s LSTM** to validate correctness.
* Add visualizations for gate activations and cell states.

---

### 👨‍💻 Author

**Prakhar Pathak** — Machine Learning Enthusiast & Developer

> “I built this project to truly understand what happens behind the black box of modern deep learning libraries.”

---

### 🌟 Why This Project Matters

Implementing an LSTM from scratch is **a rare and impressive skill** — it requires translating mathematical equations into working code while handling sequence dependencies and non-linear activations.

This project demonstrates:

* Strong grasp of **neural network fundamentals**
* Ability to **translate theory into code**
* Curiosity and perseverance — key qualities for any AI/ML engineer

---

### 🧩 Run It Yourself

```bash
# Clone this repository
git clone https://github.com/yourusername/LSTM-From-Scratch.git
cd LSTM-From-Scratch

# Open the notebook
jupyter notebook Make_copy.ipynb
```

---

### 🏁 License

This project is open-source under the **MIT License**.
Feel free to use it for learning, modification, or as a base for advanced experiments.
