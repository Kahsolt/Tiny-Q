#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/15 

from typing import Dict

import numpy as np
np.set_printoptions(precision=4, suppress=True)
np.seterr(divide='ignore', invalid='ignore')

EPS = 1e-6


class State:

  ''' represents a state vector of a quantum system: might be one qubit, two qubits or more ... '''

  def __init__(self, v: np.ndarray):
    v = np.asarray(v, dtype=np.complex64)

    assert isinstance(v, np.ndarray), 'state vector should be np.ndarray type'
    assert len(v.shape) == 1, 'state vector should be 1-dim array'
    assert np.log2(v.shape[0]) % 1 == 0.0, 'state vector length should be power of 2'

    self.v = v

  @property
  def shape(self):
    return self.v.shape

  @property
  def n_qubits(self) -> int:
    return int(np.log2(self.v.shape[0]))

  @property
  def is_pure(self) -> bool:
    ''' pure state: trace of density matrix equals to 1.0, or rho == rho**2 '''
    return np.abs(self.trace) >= 1.0 - EPS

  def __str__(self):
    return str(self.v)

  def __repr__(self):
    return repr(self.v)

  @classmethod
  def zero(cls, n:int=1):
    assert n >= 1, 'n_qubit should >= 1'
    assert isinstance(n, int), 'n_qubit should be int'
    v = [0] * 2**n ; v[0] = 1
    return cls(v)

  @classmethod
  def one(cls, n:int=1):
    assert n >= 1, 'n_qubit should >= 1'
    assert isinstance(n, int), 'n_qubit should be int'
    v = [0] * 2**n ; v[-1] = 1
    return cls(v)

  def __eq__(self, other):
    ''' v0 == v1: state equality ignores the global phase '''
    assert isinstance(other, State), f'other should be a State, but got {type(other)}'
    c = self.v / other.v            # assume c * |v> = |w>, where 'c' is a complex number
    vals = c[~np.isnan(c)]          # if vals is consistent to only one value, then 'c' is a valid global phase
    n_vals = len(vals)
    for i in range(0, n_vals-1):    # pairwise modulo diff should < EPS, if values are consistent
      for j in range(i+1, n_vals):
        if np.abs(vals[i] - vals[j]) > EPS:
          return False
    return True

  def __matmul__(self, other):
    ''' v0 @ v1 = |0>|1> = |01>: tensor product of two quantum systems '''
    assert isinstance(other, State), f'other should be a State, but got {type(other)}'
    return State(np.kron(self.v, other.v))

  def __gt__(self, other) -> str:
    ''' v0 > Measure: measure prob '''
    assert other is Measure, f'other must be Measure, but get {type(other)}({other})'
    return np.random.choice(a=self._cstates, replace=True, p=self.prob)

  def measure(self, n=1000) -> Dict[str, int]:
    results = {stat: 0 for stat in self._cstates}
    for _ in range(n):
      results[self > Measure] += 1
    return results

  @property
  def amp(self) -> np.ndarray:
    return np.abs(self.v)

  @property
  def prob(self) -> np.ndarray:
    return self.amp ** 2

  @property
  def density(self):
    ''' density matrix: rho := |v><v|
          - diag(rho) indicates probability of each classic state it'll collapses into after measurement
          - non-diag(rho) indicates **superpositioness** of the state, a pure mixed state is a simple diagonal matrix, 
            non-diagonal cells are all zeros showing that no any superpositioness
          - whether rho can be decomposed into tensor product of several smaller matrices indicates **entanglementness** of a multi-body system
    '''
    return np.outer(self.v, self.v)

  @property
  def trace(self):
    ''' trace of density matrix '''
    return np.trace(self.density)
  
  @property
  def _cstates(self):
    return [bin(x)[2:].rjust(self.n_qubits, '0') for x in range(2**self.n_qubits)]

  def info(self, title='|phi>'):
    print(title)
    print('  state:', self)
    print('  amp:', self.amp)
    print('  prob:', self.prob)
    print('  density:', self.density)
    print('  trace:', self.trace)
    print()

  def plot_prob(self, title='prob dist'):
    import matplotlib.pyplot as plt

    plt.clf()
    plt.bar(self._cstates, self.prob, color='royalblue', alpha=0.9)
    plt.ylim((0.0, 1.0))
    if title: plt.suptitle(title)
    plt.tight_layout()
    plt.show()

  def plot_density(self, title='density'):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.clf()
    sns.heatmap(np.abs(self.density), annot=True, vmin=0, vmax=1, cbar=True, cmap='Blues', alpha=0.9)
    if title: plt.suptitle(title)
    plt.tight_layout()
    plt.show()


v0 = State.zero()
v1 = State.one()


class Gate:

  ''' represents a unitary transform, aka, a quantum gate matrix '''

  def __init__(self, v):
    v = np.asarray(v, dtype=np.complex64)

    assert isinstance(v, np.ndarray), 'gate matrix should be np.ndarray type'
    assert len(v.shape) == 2, 'gate matrix should be 2-dim array'
    assert v.shape[0] == v.shape[1], 'gate matrix should be square'
    assert np.log2(v.shape[0]) % 1 == 0.0, 'gate matrix size should be power of 2'
    assert self._is_unitary(v), f'gate matrix should be unitary: {v}'

    self.v = v

  @staticmethod
  def _is_unitary(v: np.ndarray) -> bool:
    return np.abs(np.matmul(v, v.conj().T) - np.eye(2**int(np.log2(v.shape[0])))).max() < EPS

  @property
  def shape(self):
    return self.v.shape

  @property
  def n_qubits(self) -> int:
    return int(np.log2(self.v.shape[0]))

  @property
  def is_unitary(self) -> bool:
    ''' unitary: dot(A, A.dagger) == dot(A.dagger, A) = I '''
    return self._is_unitary(self.v)

  @property
  def is_hermitian(self) -> bool:
    ''' hermitian: A.dagger == A '''
    return np.abs(self.v.conj().T - self.v).max() < EPS

  def __str__(self):
    return str(self.v)

  def __repr__(self):
    return repr(self.v)

  def __eq__(self, other):
    assert isinstance(other, Gate), f'other should be a Gate, but got {type(other)}'

    if self.n_qubits > 1 and other is I:   # auto broadcast
      gate = other
      for _ in range(1, self.n_qubits):
        gate = gate @ other
    else:
      assert self.n_qubits == other.n_qubits, f'qubit count mismatch {self.n_qubits} != {other.n_qubits}'
      gate = other

    return np.abs(self.v - gate.v).max() < EPS

  def __pow__(self, pow: float):
    ''' H ** pow: gate self-power '''
    assert isinstance(pow, [float, int]), f'pow must be numerical but got {type(pow)}'
    return Gate(np.linalg.matrix_power(self.v, pow))

  def __mul__(self, other):
    ''' H * X = HX: compose two unitary transforms up '''
    assert isinstance(other, Gate), f'other must be a State, but got {type(other)}'
    assert self.n_qubits == other.n_qubits, f'qubit count mismatch {self.n_qubits} != {other.n_qubits}'
    return Gate(self.v @ other.v)

  def __matmul__(self, other):
    ''' H @ X: tensor product of two quantum gate '''
    assert isinstance(other, Gate), 'other should be a Gate'
    return Gate(np.kron(self.v, other.v))

  def __or__(self, other: State) -> State:
    ''' H | v0 = H|0>: apply this unitary transform on a state '''
    assert isinstance(other, State), f'other must be a State, but got {type(other)}'

    if self.n_qubits == 1 and other.n_qubits > 1:   # single-qubit gate auto broadcast
      gate = self
      for _ in range(1, other.n_qubits):
        gate = gate @ self
    else:
      assert self.n_qubits == other.n_qubits, f'qubit count mismatch {self.n_qubits} != {other.n_qubits}'
      gate = self

    return State(gate.v @ other.v)

  def info(self, title='|U|'):
    print(title)
    print(self)
    print()


# https://en.wikipedia.org/wiki/List_of_quantum_logic_gates
I = Gate([
  [1, 0],
  [0, 1],
])
X = NOT = Gate([
  [0, 1],
  [1, 0],
])
Y = Gate([
  [0, -1j],
  [1j, 0],
])
Z = Gate([    # Z = P(pi)
  [1,  0],
  [0, -1],
])
H = Gate(np.asarray([
  [1,  1],
  [1, -1],
]) / np.sqrt(2))
P = lambda phi: Gate([
  [1, 0],
  [0, np.e**(phi*1j)],
])
S = Gate([    # S = Z**(1/2) = P(pi/2)
  [1, 0],
  [0, np.e**(np.pi/2*1j)],
])
T = Gate([    # T = S**(1/2) = Z**(1/4)
  [1, 0],
  [0, np.e**(np.pi/4*1j)],
])
RX = lambda theta: Gate([
  [np.cos(theta/2), -1j*np.sin(theta/2)],
  [-1j*np.sin(theta/2), np.cos(theta/2)],
])
RY = lambda theta: Gate([
  [np.cos(theta/2), -np.sin(theta/2)],
  [np.sin(theta/2),  np.cos(theta/2)],
])
RZ = lambda theta: Gate([
  [np.exp(-theta/2*1j), 0],
  [0, np.exp(theta/2*1j)],
])
U = lambda theta, phi, lmbd: Gate([
  [np.cos(theta/2), -np.exp(-1j*lmbd)*np.sin(theta/2)],
  [np.exp(-1j*phi)*np.sin(theta/2), np.exp(1j*(lmbd+phi))*np.cos(theta/2)],
])
SWAP = Gate([
  [1, 0, 0, 0],
  [0, 0, 1, 0],
  [0, 1, 0, 0],
  [0, 0, 0, 1],
])
iSWAP = Gate([
  [1,  0,  0, 0],
  [0,  0, 1j, 0],
  [0, 1j,  0, 0],
  [0,  0,  0, 1],
])
CNOT = Gate([
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 0, 1],
  [0, 0, 1, 0],
])
CCNOT = Toffoli = Gate([
  [1, 0, 0, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 1, 0],
])

Measure = object()
