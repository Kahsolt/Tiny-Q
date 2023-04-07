#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/07 

from tiny_q import *

# The four-qubits HHL algorith solving linear equations in a quantum manner
#   - https://en.wikipedia.org/wiki/Quantum_algorithm_for_linear_systems_of_equations
#   - https://arxiv.org/abs/1110.2232

# case of the special: `A12` is the original essay example, `b1` is the test case from Qiskit code
# NOTE: Aij is required to be hermitian (not needed to be unitary though)
r2 = np.sqrt(2)
pi = np.pi
A12 = np.asarray([    # eigval: 2, 1
  [3, 1],             # eigvec: [0.70710678, 0.70710678], [-0.70710678, 0.70710678]
  [1, 3],
]) / 2
A13 = np.asarray([    # eigval: 3, 1
  [2, 1],             # eigvec: [0.70710678, 0.70710678], [-0.70710678, 0.70710678]
  [1, 2],
])
A23 = np.asarray([    # eigval: 3, 2
  [5, 1],             # eigvec: [0.70710678, 0.70710678], [-0.70710678, 0.70710678]
  [1, 5],
]) / 2
b0 = np.asarray([1, 0])         # |b> = |0>, for basis test
b1 = np.asarray([0, 1])         # |b> = |1>
bp = np.asarray([1,  1]) / r2   # |b> = |+>
bn = np.asarray([1, -1]) / r2   # |b> = |->

# case of the target `question1()`
# svd(A) = P * D *Q
#   [-1 0][1    0][-1 -1]
#   [ 0 1][0 1/r2][ 1 -1]
A = np.asarray([     # eigval: 1.34463698, -1.05174376
  [   1,     1],     # eigvec: [0.9454285, 0.32582963], [-0.43812257, 0.89891524]
  [1/r2, -1/r2],
])
b = np.asarray([
    1/2, 
  -1/r2,
])


print('Ablation on r')
for r in np.linspace(0.01, 1, 9):
  q = HHL(A12, b1, r=r)
  q.plots(f'r = {r}')

print('Ablation on t')
for t in np.linspace(0, 4*np.pi, 10):
  q = HHL(A12, b1, t0=t, r=1)
  q.plots(f't = {t}')
