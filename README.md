# 1D Ising Fusion Category Model – Exact Diagonalization & CFT Spectrum

This repository contains Julia code for numerical simulations of the **1D Quantum Ising Fusion Category Model** using **Exact Diagonalization (ED)**. 

By exploiting both spatial translation (momentum sector $k$) and the full **Ising fusion categorical symmetry**, the code performs complete block-diagonalization of the Hamiltonian. This allows high-precision extraction of low-lying energy spectra, direct identification of state quantum numbers (symmetry irreps), and observation of characteristic **Conformal Field Theory (CFT)** operator content and tower structures.

---


## 📐 Physics Background

The Ising fusion category consists of three simple objects/labels: **$\mathbf{1}$** (vacuum), **$\sigma$** (spin/anyonic excitation), and **$\psi$** (fermion), governed by the non-trivial fusion rule:
$$\sigma \times \sigma = \mathbf{1} + \psi$$

By constructing the Hamiltonian on anyonic chains (or topological qubit Hilbert spaces) invariant under the fusion category $F$-matrices and $R$-matrices, the critical point maps to the $c = 1/2$ minimal model CFT (Ising CFT). The low-energy spectrum $E_i(L)$ on a periodic chain of size $L$ exhibits universal finite-size scaling:

$$E_i(L) - E_0(L) \approx \frac{2\pi v}{L} (h_i + \bar{h}_i + n_i + \bar{n}_i)$$

where $h_i, \bar{h}_i$ are the conformal weights of the corresponding primary operator, and $n_i, \bar{n}_i \in \mathbb{N}$ label descendant levels.

---
