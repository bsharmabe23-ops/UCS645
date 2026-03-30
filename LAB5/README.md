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

```bash
mpicc filename.c -o filename
mpirun --oversubscribe -np 4 ./filename
```

---

# 🔹 Q1: DAXPY Operation

## Problem

Perform the operation:
X[i] = a * X[i] + Y[i]

## Approach

* The data is divided among multiple processes.
* Each process performs calculations on its part.
* Execution happens in parallel.

## Output

```
Process 1 time = 0.000044
Process 0 time = 0.000045
Process 2 time = 0.000066
Process 3 time = 0.000040
```

## Conclusion

Parallel execution reduces computation time for vector operations.

---

# 🔹 Q2: Broadcast Communication

## Problem

Compare manual communication with MPI_Bcast.

## Approach

* First, data is sent manually using MPI_Send and MPI_Recv.
* Then, MPI_Bcast is used.
* Execution time is measured in both cases.

## Output

```
Manual Broadcast Time = 0.025857
MPI_Bcast Time = 0.002140
```

## Conclusion

MPI_Bcast is faster and more efficient than manual communication.

---

# 🔹 Q3: Dot Product using MPI_Reduce

## Problem

Compute dot product of two vectors.

## Approach

* Each process computes part of the result.
* MPI_Reduce combines all results.

## Output

```
Dot Product = 2000000.000000
Time = 0.001149
```

## Conclusion

MPI_Reduce efficiently combines results from multiple processes.

---

# 🔹 Q4: Prime Numbers

## Problem

Check whether numbers are prime.

## Approach

* A number is checked for divisibility up to √n.
* Work is divided among processes.

## Output

```
2 is Prime
3 is Prime
4 is Not Prime
5 is Prime
6 is Not Prime
7 is Prime
8 is Not Prime
9 is Not Prime
10 is Not Prime
```

## Conclusion

Parallel execution allows checking multiple numbers simultaneously.

---

# 🔹 Q5: Perfect Numbers

## Problem

Find perfect numbers.

## Approach

* Sum of divisors is calculated.
* If sum equals the number → perfect number.
* Work is divided among processes.

## Output

```
Process 0: 6 is Perfect
Process 2: 28 is Perfect
Process 2: 496 is Perfect
```

## Conclusion

Parallel processing improves efficiency for large computations.

---

## 🔹 Concepts Learned

* MPI basics (rank, size)
* Blocking vs Non-blocking communication
* MPI_Bcast and MPI_Reduce
* Parallel execution using multiple processes

---

## 🔹 Result

All programs (Q1 to Q5) were successfully compiled and executed.

---

## 🔹 Final Conclusion

This lab helped in understanding how parallel programming works using MPI.
Multiple processes can work together to complete tasks faster and more efficiently.

