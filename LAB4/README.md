
# Assignment 4 – Introduction to MPI

Course: UCS645 – Parallel & Distributed Computing  
Topic: Message Passing Interface (MPI)  
Number of Processes Used: 4  
MPI Implementation: OpenMPI  
Execution Environment: Ubuntu (WSL)

---

## 1. Introduction

Message Passing Interface (MPI) is a standardized and portable message-passing system designed for parallel programming on distributed memory architectures. MPI enables multiple processes to execute simultaneously and communicate efficiently.

This assignment demonstrates the following MPI concepts:

- Process ranks and communicators
- Point-to-point communication
- Collective communication
- Data distribution using MPI_Scatter
- Data aggregation using MPI_Reduce
- Location-based reduction using MPI_MAXLOC and MPI_MINLOC
- Parallel numerical computation

All programs were compiled using `mpicc` and executed using 4 parallel processes.

---

## 2. Compilation and Execution

### Compilation

```bash
mpicc -o program_name program_name.c
```

### Execution (WSL with OpenMPI)

```bash
mpirun --oversubscribe -np 4 ./program_name
```

Note: The `--oversubscribe` option is required in WSL environments when the number of processes exceeds available CPU cores.

---

# Exercise 1: Ring Communication

## Problem Description

- Process 0 initializes value = 100  
- Each process adds its rank to the value  
- The value circulates in a ring topology  
- The final value returns to Process 0  

Ring topology logic:

```
next = (rank + 1) % size
prev = (rank - 1 + size) % size
```

## MPI Concepts Used

- MPI_Init
- MPI_Comm_rank
- MPI_Comm_size
- MPI_Send
- MPI_Recv

## Execution Output (4 Processes)

```
Process 3 updated value to: 106
Process 0 starts with value: 100
Process 0 received final value: 106
Process 1 updated value to: 101
Process 2 updated value to: 103
```

Note: Output order may vary due to parallel execution.

## Verification

Final value calculation:

```
100 + (1 + 2 + 3) = 106
```

Correct circular communication is verified.

---

# Exercise 2: Parallel Array Sum

## Problem Description

- Create an array of size 100 with values from 1 to 100 in Process 0  
- Distribute array segments using MPI_Scatter  
- Each process computes the sum of its segment  
- Use MPI_Reduce to compute the global sum  
- Compute and print the average  

## MPI Concepts Used

- MPI_Scatter
- MPI_Reduce (MPI_SUM)
- Parallel workload distribution

## Execution Output

```
Global Sum = 5050
Average = 50.50
```

## Mathematical Verification

Sum formula:

```
n(n + 1) / 2
100 × 101 / 2 = 5050
```

Average:

```
5050 / 100 = 50.5
```

The computed result matches the expected mathematical result.

---

# Exercise 3: Global Maximum & Minimum

## Problem Description

- Each process generates 10 random numbers (0–1000)  
- Each process computes its local maximum and minimum  
- Use MPI_MAXLOC and MPI_MINLOC  
- Identify both the global value and the process containing it  

## MPI Concepts Used

- MPI_Reduce
- MPI_MAXLOC
- MPI_MINLOC
- Struct-based reduction (value + rank)

## Execution Output

```
Global Maximum: 979 (Process 3)
Global Minimum: 75 (Process 1)
```

Note: Values vary across runs due to random number generation.

## Observation

Correct identification of both value and corresponding process confirms proper use of location-based reduction operations.

---

# Exercise 4: Parallel Dot Product

## Problem Description

Compute the dot product of:

```
A = [1,2,3,4,5,6,7,8]
B = [8,7,6,5,4,3,2,1]
```

Expected Result: 120

## MPI Concepts Used

- MPI_Scatter
- Local computation
- MPI_Reduce (MPI_SUM)

## Execution Output

```
Dot Product = 120
```

## Verification

```
1×8 + 2×7 + 3×6 + 4×5 + 5×4 + 6×3 + 7×2 + 8×1 = 120
```

The computed result matches the expected value.

---

# Performance Considerations

Execution time in MPI programs can be measured using:

```
MPI_Wtime()
```

Speedup:

```
S_p = T1 / Tp
```

Efficiency:

```
E_p = S_p / p
```

Where:
- T1 = Execution time using 1 process
- Tp = Execution time using p processes

Efficiency indicates how effectively parallel resources are utilized.

---

# Observations

- MPI enables scalable and structured parallel programming.
- Output ordering may differ due to concurrent execution.
- Collective operations significantly simplify distributed data management.
- Communication overhead is minimal for small problem sizes.
- Performance improvement depends on workload size and number of processes.

---

# Conclusion

This assignment successfully demonstrated:

- Point-to-point communication
- Collective communication
- Parallel data distribution
- Reduction operations
- Location-based reductions
- Parallel numerical computation

All programs executed successfully using 4 processes and produced correct expected results, validating the correct implementation of MPI concepts.
