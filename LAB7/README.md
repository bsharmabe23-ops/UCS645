# 🚀 Assignment 7: CUDA Part II

**Course:** Parallel Computing  
**Student:** Bipasha Sharma  
**Institute:** Thapar Institute of Engineering & Technology  

---

## 📁 Project Structure

CUDA_Assignment7/
├── problem1.cu     
├── problem2.cu     
├── problem3.cu     
└── README.md  

---

## ⚙️ Environment

- **Compiler:** NVCC (CUDA Toolkit)  
- **Operating System:** Ubuntu 24.04 (WSL2)  
- **Language:** CUDA C++  

---

## 🧮 Problem 1: Sum of First N Integers (N = 1024)

### 🔍 Description
A single CUDA kernel uses two threads performing different tasks:

- **Thread 0:** Computes sum iteratively (1 → N)  
- **Thread 1:** Computes sum using formula → `N × (N + 1) / 2`  

### 📌 Key Concepts
- `threadIdx.x` for task division  
- `cudaMalloc` for GPU memory allocation  
- `cudaMemcpy` for CPU ↔ GPU transfer  
- Branching within kernel based on thread ID  

### 📊 Output
