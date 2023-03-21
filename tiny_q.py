#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/15 

from __future__ import annotations
from typing import List, Dict, Union, Callable, Any

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import fractional_matrix_power
import numpy as np
np.set_printoptions(precision=4, suppress=True)
np.seterr(divide='ignore', invalid='ignore')

DTYPE = np.complex64
EPS   = 1e-6

if 'syntax hijack':
  from numpy import pi, sin, cos

  class float(float):
    def __xor__(self, other):
      return other.__rpow__(self)

  class array(np.ndarray):
    @property
    def dagger(self):
      return self.conj().T

  e = float(np.e)
  i = np.complex64(0 + 1j)    # imaginary unit


class Meta:

  ''' represents a 1d vector or 2d tensor '''

  Null = None     # empty system containing 0 qubits

  def __init__(self, v:Union[np.ndarray, list]):
    x = np.asarray(v, dtype=DTYPE)
    self.v = array(x.shape, buffer=x, dtype=x.dtype)

  def __str__(self) -> str:
    return str(self.v)

  def __repr__(self) -> str:
    return repr(self.v)

  @property
  def shape(self):
    return self.v.shape

  @property
  def n_qubits(self) -> int:
    return int(np.log2(self.v.shape[0]))

  @property
  def dagger(self) -> Meta:
    return self.__class__(self.v.dagger)

  def __matmul__(self, other: Union[Meta, Meta.Null, int]) -> Union[State, Gate]:
    '''
      Meta: tensor product of sub-systems
        - v0 @ v1 = |0>|1> = |01>: tensor product of two quantum systems
        - H @ X: tensor product of two quantum gate
      int: tensor product by self n_times
        - v0 @ 3 = v('000')
        - H @ 3 = H @ H @ H
    '''

    if isinstance(other, int):
      if other == 0: return Meta.Null
      r = self
      for _ in range(1, other):
        r = r @ self
      return r
  
    if other is Meta.Null: return self
    assert isinstance(other, (State, Gate)), 'other should be a State or Gate, but got {type(other)}'
    return self.__class__(np.kron(self.v, other.v))

  def __rmatmul__(self, other: Meta.Null) -> Union[State, Gate]:
    assert other is Meta.Null, f'other should be Meta.Null, but got {type(other)}'
    return self


class State(Meta):

  ''' represents a state vector of a quantum system: might be one qubit, two qubits or more ... '''

  def __init__(self, v):
    super().__init__(v)

    assert isinstance(self.v, np.ndarray), 'state vector should be np.ndarray type'
    assert len(self.shape) == 1, 'state vector should be 1-dim array'
    assert np.log2(self.shape[0]) % 1 == 0.0, 'state vector length should be power of 2'

  @classmethod
  def zero(cls, n:int=1) -> State:
    assert n >= 1, 'n_qubit should >= 1'
    assert isinstance(n, int), 'n_qubit should be int'
    v = [0] * 2**n ; v[0] = 1
    return cls(v)

  @classmethod
  def one(cls, n:int=1) -> State:
    assert n >= 1, 'n_qubit should >= 1'
    assert isinstance(n, int), 'n_qubit should be int'
    v = [0] * 2**n ; v[-1] = 1
    return cls(v)

  def __eq__(self, other: Any) -> bool:
    ''' v0 == v1: state equality ignores the global phase '''
    if not isinstance(other, State): raise NotImplemented

    c = self.v / other.v            # assume c * |v> = |w>, where 'c' is a complex number
    vals = c[~np.isnan(c)]          # if vals is consistent to only one value, then 'c' is a valid global phase
    n_vals = len(vals)
    for j in range(0, n_vals-1):    # pairwise modulo diff should < EPS, if values are consistent
      for k in range(j+1, n_vals):
        if np.abs(vals[j] - vals[k]) > EPS:
          return False
    return True

  def __lt__(self, other: Measure):
    '''
      v0 < Measure: project measure by computational basis, then make quantum state collapse **inplace**
        - the state shows probability of p(i) = <phi|Mi|phi> to collapse on \frac{Mi|phi>}{sqrt(p(i))}
        - for projection measure on computational basis, you just got the i-th cstate |i>
    '''
    assert other is Measure, f'other must be Measure, but got {other}'
    self.v = v(self > Measure).v

  def __gt__(self, other: Union[Measure, State, MeasureOp]) -> Union[str, Dict[str, int], float]:
    '''
      v0 > Measure: project measure by computational basis, return result as binary string
      v0 > Measure(n): project measure by computational basis, return results as a Dict[str, int]
      v0 > MeasureOp|State: project measure by given measure operator Mi or state |psi>, return the collapse-to probability 
    '''

    if other is Measure:
      ''' one-shot measure '''
      return np.random.choice(a=self._cstates, replace=True, p=self.prob)
    elif isinstance(other, Callable):
      ''' Monte-Carlo sample '''
      n = other()
      assert isinstance(n, int), f'count of Measure must be int type, but got {type(n)}'
      results = {stat: 0 for stat in self._cstates}
      for _ in range(n):
        results[self > Measure] += 1
      return results
    elif isinstance(other, State):
      ''' p = |<phi|psi>|^2 '''
      return np.abs(np.inner(self.v, other.v)) ** 2
    elif isinstance(other, MeasureOp):
      ''' p(i) = <phi| Mi.dagger * Mi |phi> '''
      return np.abs(self.dagger.v @ other.dagger.v @ other.v @ self.v)
    else:
      raise TypeError(f'other should be a MeasureOp or a State, or the Measure object, but got {type(other)}({other})')

  @property
  def is_pure(self) -> bool:
    '''
      rho properties for pure state: 
        - idempotent: rho^2 == rho
        - tr(rho) = Σi <i|rho|i> = 1
        - tr(rho^2) == 1
        - hermitian: rho.dagger = rho
        - positive semi-definite: <phi|rho|phi> >= 0
    '''
    return np.trace(np.linalg.matrix_power(self.density, 2)) >= 1.0 - EPS

  @property
  def amp(self) -> np.ndarray:
    ''' |phi> = Σi αi|i>, amplitude of i-th basis amp(|i>) = abs(αi) '''
    return np.abs(self.v)

  @property
  def prob(self) -> np.ndarray:
    ''' |phi> = Σi αi|i>, probability of i-th basis prob(|i>) = |αi|^2 '''
    return self.amp ** 2

  @property
  def density(self) -> np.ndarray:
    '''
      rho := |phi><phi| (pure state) or Σi αi|phi_i><phi_i| (mixed state), density matrix
        - diag(rho) indicates probability of each classic state it'll collapses into after measurement
        - non-diag(rho) indicates **superpositioness** of the state, a pure mixed state is a simple diagonal matrix, 
          non-diagonal cells are all zeros showing that no any superpositioness
        - whether rho can be decomposed into tensor product of several smaller matrices indicates **entanglementness** of a multi-body system
      NOTE: one density matrix corresponds to many quantum states respect to a global phase
    '''
    return np.outer(self.v, self.v)

  @property
  def trace(self) -> float:
    ''' tr(rho) = Σ diag(rho): trace of density matrix '''
    return np.trace(self.density)

  @property
  def _cstates(self) -> List[str]:
    return [bin(x)[2:].rjust(self.n_qubits, '0') for x in range(2**self.n_qubits)]

  def info(self, title='|phi>'):
    print(title)
    print('  state:', self)
    print('  amp:', self.amp)
    print('  prob:', self.prob)
    print('  density:', self.density)
    print('  trace:', self.trace)
    print()

  def plot_prob(self, title='prob'):
    plt.clf()
    plt.bar(self._cstates, self.prob, color='royalblue', alpha=0.9)
    plt.ylim((0.0, 1.0))
    if title: plt.suptitle(title)
    plt.tight_layout()
    plt.show()

  def plot_density(self, title='rho'):
    plt.clf()
    sns.heatmap(np.abs(self.density), annot=True, vmin=0, vmax=1, cbar=True, cmap='Blues', alpha=0.9)
    if title: plt.suptitle(title)
    plt.tight_layout()
    plt.show()

  def plots(self, title='|phi>'):
    plt.clf()
    plt.subplot(121)
    plt.title('prob')
    plt.bar(self._cstates, self.prob, color='royalblue', alpha=0.9)
    plt.ylim((0.0, 1.0))
    plt.subplot(122)
    plt.title('density')
    sns.heatmap(np.abs(self.density), annot=True, vmin=0, vmax=1, cbar=True, cmap='Blues', alpha=0.9)
    if title: plt.suptitle(title)
    plt.tight_layout()
    plt.show()


class Gate(Meta):

  ''' represents a unitary transform, aka, a quantum gate matrix '''

  def __init__(self, v):
    super().__init__(v)

    assert isinstance(self.v, np.ndarray), 'gate matrix should be np.ndarray type'
    assert len(self.shape) == 2, 'gate matrix should be 2-dim array'
    assert self.shape[0] == self.shape[1], 'gate matrix should be square'
    assert np.log2(self.shape[0]) % 1 == 0.0, 'gate matrix size should be power of 2'
    assert self.is_unitary, f'gate matrix should be unitary: {self.v}'

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, Gate): raise NotImplemented

    if self.n_qubits > 1 and other is I:   # auto broadcast
      other = other @ self.n_qubits
    else:
      assert self.n_qubits == other.n_qubits, f'qubit count mismatch {self.n_qubits} != {other.n_qubits}'

    return np.abs(self.v - other.v).max() < EPS

  def __neg__(self) -> Gate:
    ''' Ph(-pi)*U == -U '''
    return Ph(-pi) * self

  def __pow__(self, pow: float):
    ''' H**pow: gate self-power '''
    return self.__xor__(pow)

  def __xor__(self, pow: float):
    ''' H^pow: gate self-power '''
    assert isinstance(pow, (float, int)), f'pow should be numerical type but got {type(pow)}'
    if isinstance(pow, int):
      return Gate(np.linalg.matrix_power(self.v, pow))
    else:
      return Gate(fractional_matrix_power(self.v, pow))

  def __mul__(self, other: Gate) -> Gate:
    ''' H * X = HX: compose two unitary transforms up '''
    assert isinstance(other, Gate), f'other should be a State, but got {type(other)}'
    assert self.n_qubits == other.n_qubits, f'qubit count mismatch {self.n_qubits} != {other.n_qubits}'
    return Gate(self.v @ other.v)

  def __or__(self, other: State) -> State:
    ''' H | v0 = H|0>: apply this unitary transform on a state '''
    assert isinstance(other, State), f'other should be a State, but got {type(other)}'

    if self.n_qubits == 1 and other.n_qubits > 1:   # single-qubit gate auto broadcast
      self = self @ other.n_qubits
    else:
      assert self.n_qubits == other.n_qubits, f'qubit count mismatch {self.n_qubits} != {other.n_qubits}'

    return State(self.v @ other.v)

  @property
  def is_unitary(self) -> bool:
    ''' unitary: dot(A, A.dagger) == dot(A.dagger, A) = I '''
    return np.abs(np.matmul(self.v, self.v.dagger) - np.eye(2**self.n_qubits)).max() < EPS

  @property
  def is_hermitian(self) -> bool:
    ''' hermitian: A.dagger == A '''
    return np.abs(self.v.dagger - self.v).max() < EPS

  def info(self, title='|U|'):
    print(title)
    print(self)
    print()


class MeasureOp(Meta):

  ''' represents a partial measurement operator from some set '''

  def __init__(self, v):
    super().__init__(v)

    assert isinstance(self.v, np.ndarray), 'measure operator should be np.ndarray type'
    assert len(self.shape) == 2, 'measure operator should be 2-dim array'
    assert self.shape[0] == self.shape[1], 'measure operator should be square'
    assert np.log2(self.shape[0]) % 1 == 0.0, 'measure operator size should be power of 2'

  @staticmethod
  def check_completeness(ops: List[MeasureOp]) -> bool:
    ''' Σi (Mi.dagger * Mi) = I: completeness equation for a measure operator set '''

    if not ops: return False
    for op in ops: assert isinstance(op, MeasureOp), f'elem of ops should be a MeasureOp, but got {type(op)}'

    s = np.zeros_like(ops[0].v)
    for Mi in ops:
      s += Mi.dagger.v @ Mi.v
    s -= np.eye(2**ops[0].n_qubits)
    return np.abs(s).max() < EPS


def v(string: str) -> State:
  assert string, 'string should not be empty'
  qubits = { '0': v0, '1': v1 }
  v = qubits[string[0]]
  for c in string[1:]:
    v = v @ qubits[c]
  return v


# https://en.wikipedia.org/wiki/List_of_quantum_logic_gates
I = Gate([                    # indentity
  [1, 0],
  [0, 1],
])
Ph = lambda theta: Gate([     # alter global phase, e^(-i*theta*I)
  [e^(-i*theta), 0],
  [0, e^(-i*theta)],
])
X = NOT = Gate([              # flip amplitude
  [0, 1],
  [1, 0],
])
Y = Gate([                    # flip amplitude & flip phase by imag
  [0, -i],
  [i,  0],
])
Z = Gate([                    # flip phase by real, Z = P(pi)
  [1,  0],
  [0, -1],                    # e^(i*pi) == -1
])
H = Gate(np.asarray([         # scatter basis, make superposition
  [1,  1],
  [1, -1],
]) / np.sqrt(2))
SX = V = Gate(np.asarray([    # sqrt(X)
  [1 + i, 1 - i],
  [1 - i, 1 + i],
]) / 2)
P = lambda phi: Gate([        # alter phase
  [1, 0],
  [0, e^(i*phi)],
])
S = Gate([                    # alter phase, S = Z^(1/2) = P(pi/2)
  [1, 0],
  [0, e^(i*pi/2)],           # e^(i*pi/2) == i
])
T = Gate([                    # alter phase, T = S^(1/2) = Z^(1/4) = P(pi/4)
  [1, 0],
  [0, e^(i*pi/4)],            # e^(i*pi/4) == (1+i)/sqrt(2)
])
RX = lambda theta: Gate([     # alter amplitude, e^(-i*X*theta/2)
  [cos(theta/2), -i*sin(theta/2)],
  [-i*sin(theta/2), cos(theta/2)],
])
RY = lambda theta: Gate([     # alter amplitude, e^(-i*Y*theta/2)
  [cos(theta/2), -sin(theta/2)],
  [sin(theta/2),  cos(theta/2)],
])
RZ = lambda theta: Gate([     # alter phase, e^(-i*Z*theta/2)
  [e^(-i*theta/2), 0],
  [0, e^(i*theta/2)],
])
RZ1 = lambda theta: Gate([    # another form of RZ except a g_phase of e^(i*theta/2)
  [1, 0],
  [0, e^(i*theta)],
])
U = lambda theta, phi, lmbd: Gate([
  [cos(theta/2), -e^(-i*lmbd)*sin(theta/2)],
  [e^(-i*phi)*sin(theta/2), e^(i*(lmbd+phi))*cos(theta/2)],
])
U1 = lambda alpha, beta, gamma, delta: Ph(alpha) * RZ(beta) * RY(gamma) * RZ(delta)     # universal Z-Y decomposition
SWAP = Gate([
  [1, 0, 0, 0],
  [0, 0, 1, 0],
  [0, 1, 0, 0],
  [0, 0, 0, 1],
])
iSWAP = Gate([                # SWAP while flip relative phase 
  [1, 0, 0, 0],
  [0, 0, i, 0],
  [0, i, 0, 0],
  [0, 0, 0, 1],
])
fSWAP = Gate([
  [1, 0, 0,  0],
  [0, 0, 1,  0],
  [0, 1, 0,  0],
  [0, 0, 0, -1],
])
CNOT = CX = Gate([            # make entanglement
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 0, 1],
  [0, 0, 1, 0],
])
DCNOT = Gate([
  [1, 0, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
  [0, 1, 0, 0],
])
CZ = Gate([
  [1, 0, 0,  0],
  [0, 1, 0,  0],
  [0, 0, 1,  0],
  [0, 0, 0, -1],
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

v0 = State.zero()   # |0>
v1 = State.one()    # |1>
h0 = H | v0         # |+>
h1 = H | v1         # |->
bell_state = CNOT * (H @ I) | State.zero(2)  # |00>+|11>
ghz_state = (I @ CNOT) * (CNOT @ I) * (H @ I @ I) | State.zero(3)   # |000>+|111>

Measure = lambda n=1000: (lambda: n)
M0 = MeasureOp([
  [1, 0],
  [0, 0],
])
M1 = MeasureOp([
  [0, 0],
  [0, 1],
])


if __name__ == '__main__':
  # => basic states
  assert (v0 > v1) < EPS and (v1 > v0) < EPS            # diagonal
  assert (h0 > h1) < EPS and (h1 > h0) < EPS
  for q1 in [v0, v1, h0, h1]:
    for q2 in [v0, v1, h0, h1]:
      if q1 == q2: assert (q1 > q2) - 1   < EPS
      else:        assert (q1 > q2) - 0.5 < EPS
  assert v('0') == v0 and v('1') == v1
  assert v('10110') == v1 @ v0 @ v1 @ v1 @ v0
  assert v0.dagger == v0

  # => global phase gate
  assert X.dagger == X
  assert Ph(0) == I
  assert Ph(pi) == Ph(-pi) == Gate(-I.v)

  # => pauli gates
  assert X^2 == I and Y^2 == I and Z^2 == I
  assert H*Z*H == X and H*X*H == Z
  assert X*Y == -Y*X and Y*Z == -Z*Y and Z*X == -X*Z 
  assert Gate(((X*Y).v-(Y*X).v)/2) == Gate(i*Z.v)       # σiσj - σjσi = 2*i*σk, where i,j,k is a cyclic permutation of of X,Y,Z

  # => phase gates
  assert SX^2 == X
  assert T^2 == S and S^2 == Z and Z^2 == I
  assert Z == P(pi) and S == P(pi/2) and T == P(pi/4)

  # => measure operator set
  assert MeasureOp.check_completeness([M0, M1])

  from code import interact
  interact(local=globals())
