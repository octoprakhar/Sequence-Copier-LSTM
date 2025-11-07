# 🧠 LSTM from Scratch: Sequence Copy Task (Pure NumPy)

## Project Overview

This repository features a **Long Short-Term Memory (LSTM) network implemented entirely from scratch using only NumPy**. This project goes beyond using high-level deep learning APIs to demonstrate a fundamental, mathematical understanding of **Recurrent Neural Networks (RNNs)** and their complex mechanisms for handling sequential data.

The model is applied to the canonical **Sequence Copy Task**, where the network must memorize an input sequence (e.g., a binary string) and reproduce it after a delay, proving its ability to manage **long-term dependencies**.

***

## 🌟 Core Technical Achievement: LSTM Gate Mechanics

The most significant achievement of this project is the manual implementation and tracking of the **four core gating mechanisms** and two memory states across every time step ($\mathbf{t}$):

### 1. The Gates (The Brain of the LSTM)

| Gate | Function | Mathematical Role |
| :--- | :--- | :--- |
| **Forget Gate ($\mathbf{f_t}$)** | Decides which information from the previous **Cell State ($\mathbf{C_{t-1}}$)** should be discarded. | $\mathbf{f_t} = \sigma(\mathbf{W}_f \mathbf{h}_{t-1} + \mathbf{U}_f \mathbf{x}_t + \mathbf{b}_f)$ |
| **Input Gate ($\mathbf{i_t}$)** | Decides which new information from the current input ($\mathbf{x}_t$) should be stored in the Cell State. | $\mathbf{i_t} = \sigma(\mathbf{W}_i \mathbf{h}_{t-1} + \mathbf{U}_i \mathbf{x}_t + \mathbf{b}_i)$ |
| **Candidate State ($\mathbf{\tilde{C}_t}$)** | Creates a vector of *candidate* values to be added to the $\mathbf{C}_t$. | $\mathbf{\tilde{C}_t} = \text{tanh}(\mathbf{W}_C \mathbf{h}_{t-1} + \mathbf{U}_C \mathbf{x}_t + \mathbf{b}_C)$ |
| **Output Gate ($\mathbf{o_t}$)** | Controls which parts of the Cell State are used to compute the **Hidden State ($\mathbf{h_t}$)**. | $\mathbf{o_t} = \sigma(\mathbf{W}_o \mathbf{h}_{t-1} + \mathbf{U}_o \mathbf{x}_t + \mathbf{b}_o)$ |

### 2. State Management: Solving the Vanishing Gradient Problem

The project demonstrates understanding of the LSTM's solution to the vanishing gradient problem through the **Cell State ($\mathbf{C_t}$)** update.

The new Cell State is an element-wise combination of the old state (scaled by the **Forget Gate**) and the new candidate state (scaled by the **Input Gate**). This additive structure creates a "straight pipe" for gradients to flow backward through time, essential for learning long-range dependencies:

$$\mathbf{C_t} = \mathbf{f_t} \odot \mathbf{C_{t-1}} + \mathbf{i_t} \odot \mathbf{\tilde{C}_t}$$

***

## 🛠️ Implementation Details

The core logic resides in the `Copy_Lstm` class, which manages all weight matrices ($\mathbf{W}$, $\mathbf{U}$) and bias vectors ($\mathbf{b}$) for each gate.

### Key Implemented Methods:

* **`__init__`**: Initializes all **24 weight/bias parameters** manually (e.g., `self.W_f`, `self.U_f`, `self.b_f` for the forget gate).
* **`sigmoid` / `tanh`**: Implementation of the primary activation functions used in RNNs.
* **`forward()`**: The central method that iterates through the input sequence, calculates the four gates, updates the Cell and Hidden states, and generates the final prediction for each time step.
* **Data Generation:** Includes custom code to generate sequential integer data and convert it to a binary representation for the Copy Task.

### Architecture Notes:

* **Sequence-to-Sequence:** The architecture handles a sequence input and generates a sequence output of the same length (or greater, depending on the task definition).
* **Backpropagation Through Time (BPTT):** The forward pass is designed to store all intermediate values ($\mathbf{f_t}, \mathbf{i_t}, \mathbf{C_t}, \mathbf{h_t}, \text{etc.}$) necessary for the subsequent backpropagation through time calculation.

***

## ➡️ Next Steps & Further Development

To make this project production-ready, the next major challenge is implementing the full BPTT algorithm.

* **Backpropagation Implementation:** Manually calculate the gradients for the gates, the weight matrices, and the hidden/cell states, chaining them backward through the entire sequence length.
* **Gradient Stability:** Implement the full **Adam Optimizer** from scratch to manage adaptive learning rates and momentum, which is critical for stabilizing the volatile gradients encountered in BPTT.
* **Formal Verification:** Use a numerical approach (like **Gradient Checking**) to rigorously verify the correctness of the analytical BPTT implementation.

***

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `Make_copy.ipynb` | Jupyter Notebook containing the `Copy_Lstm` class definition, the data generation logic, and the training/forward pass execution. |
| `README.md` | This document. |
