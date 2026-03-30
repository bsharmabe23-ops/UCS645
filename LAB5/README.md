
# LAB 5 - MPI Programming Assignment

## Name: Bipasha Sharma

## Subject: UCS635 (Parallel Computing Lab)

---

## 🔹 Aim

The aim of this lab is to understand parallel programming using MPI (Message Passing Interface) and to implement different programs using multiple processes.

---

## 🔹 Software and Tools Used

* Ubuntu (WSL)
* VS Code
* MPI Compiler (mpicc)
* C Programming Language

---

## 🔹 How to Compile and Run Programs

For all programs, the following commands were used:

```
mpicc filename.c -o filename
mpirun --oversubscribe -np 4 ./filename
```

---

# 🔹 Q1: DAXPY Operation

## Problem

Perform the operation:
X[i] = a * X[i] + Y[i]

## Approach

* The total data is divided among all processes.
* Each process performs calculations on its own portion.
* Parallel execution reduces overall computation time.

## Output

Each process calculates part of the result and prints execution time.

## Conclusion

This program shows how parallel processing improves performance for vector operations.

---

# 🔹 Q2: Broadcast Communication

## Problem

Compare manual communication with MPI_Bcast.

## Approach

* First, data is sent manually using MPI_Send and MPI_Recv.
* Then, the same operation is performed using MPI_Bcast.
* Time is measured in both cases.

## Output

Displays:

* Manual Broadcast Time
* MPI_Bcast Time

## Conclusion

MPI_Bcast is faster because it uses optimized communication (tree-based method).

---

# 🔹 Q3: Dot Product using MPI_Reduce

## Problem

Compute the dot product of two large vectors.

## Approach

* Each process computes a part of the dot product.
* MPI_Reduce is used to combine all partial results.

## Output

* Final dot product value
* Execution time

## Conclusion

MPI_Reduce efficiently combines results from multiple processes.

---

# 🔹 Q4: Prime Numbers

## Problem

Find whether numbers are prime.

## Approach

* A number is checked for divisibility up to √n.
* Each process handles different numbers.

## Output

Example:
2 is Prime
3 is Prime
4 is Not Prime

## Conclusion

Parallel execution allows checking multiple numbers at the same time.

---

# 🔹 Q5: Perfect Numbers

## Problem

Find all perfect numbers.

## Approach

* For each number, sum of divisors is calculated.
* If sum equals the number → perfect number.
* Work is divided among processes.

## Output

Example:
6 is Perfect
28 is Perfect
496 is Perfect

## Conclusion

Parallel processing improves performance for large ranges.

---

## 🔹 Concepts Learned

* MPI basics (rank, size)
* Blocking vs Non-blocking communication
* MPI_Bcast and MPI_Reduce
* Parallel computation techniques
* Compilation and execution of MPI programs

---

## 🔹 Result

All programs (Q1 to Q5) were successfully implemented and executed.

---

## 🔹 Final Conclusion

This lab helped in understanding how parallel programming works using MPI.
Multiple processes can work together efficiently, reducing computation time and improving performance.
