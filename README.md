# Analytic Gradient Neural Network (MATLAB)

This repository demonstrates a **manually implemented artificial neural network (ANN)** trained to reproduce the **barometric pressure–altitude relationship** using **analytic gradients**.  
It is written entirely in **MATLAB**, with no deep learning toolboxes — every step of the forward pass, backpropagation, and gradient update is derived and coded from first principles.

---

## 🌍 Overview

The network learns how **atmospheric pressure decreases with altitude**, following the barometric law:

$P(z) = P_0 \exp\left(-\dfrac{M g z}{R T}\right)$

A 1–4–1 feedforward neural network (one input, four hidden neurons, one output) is trained using gradient descent with **analytic Jacobians** — enabling precise and transparent learning dynamics.

