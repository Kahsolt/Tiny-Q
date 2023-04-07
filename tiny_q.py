#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/15 

from __future__ import annotations
from typing import List, Dict, Union, Callable, Any

import os
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


''' Core '''

class Meta:

  ''' represents quantum system tensor data (1d vector or 2d tensor) '''

  Null = None     # empty system containing 0 qubits

  def __init__(self, v:Union[np.ndarray, list]):
    x = np.ascontiguousarray(v, dtype=DTYPE)
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

  @staticmethod
  def system_expansion(x:Meta, n:int) -> Meta:
    ''' x @ n: self tensor product of system x by n times '''
    assert isinstance(n, int) and n >= 0, 'n should be a non-negative integer'
    if n == 0: return Meta.Null

    r = x
    for _ in range(1, n):
      r = r @ x
    return r

  def __matmul__(self, other: Union[Meta, Meta.Null, int]) -> Union[State, Gate]:
    if isinstance(other, int):
      return Meta.system_expansion(self, other)

    if other is Meta.Null: return self
    assert isinstance(other, (State, Gate)), f'other should be a State or Gate, but got {type(other)}'
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

  def __matmul__(self, other: Union[State, int]) -> State:
    '''
      v0 @ v1 = |0>|1> = |01>: tensor product of two quantum systems
      v0 @ 3 = v('000'), tensor product by self n_times
    '''
    return super().__matmul__(other)

  def __rmatmul__(self, other: Meta.Null) -> State:
    return super().__rmatmul__(other)

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
  def cval(self) -> int:
    ''' |i> -> int(i), for classic state '''
    idx = self.v.argmax()
    val = self.v[idx]
    assert abs(val) - 1.0 < EPS, f'cannot decode cbit of a super-positioned state, the max amplitude is {val}'
    return idx

  @property
  def cbit(self) -> str:
    ''' |i> -> bin(i), for classic state '''
    return self._val_to_bit(self.cval)

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

  def _val_to_bit(self, val:int) -> str:
    return bin(val)[2:].rjust(self.n_qubits, '0')

  @property
  def _cstates(self) -> List[str]:
    return [self._val_to_bit(x) for x in range(2**self.n_qubits)]

  def info(self, title='|phi>'):
    print(title)
    print('  state:', self)
    print('  amp:', self.amp)
    print('  prob:', self.prob)
    print('  density:', self.density)
    print('  trace:', self.trace)
    print()

  def plot_prob(self, title='prob'):
    if os.environ.get('IGNORE_PLOTS'): return

    plt.clf()
    plt.bar(self._cstates, self.prob, color='royalblue', alpha=0.9)
    plt.ylim((0.0, 1.0))
    if title: plt.suptitle(title)
    plt.tight_layout()
    plt.show()

  def plot_density(self, title='rho'):
    if os.environ.get('IGNORE_PLOTS'): return

    plt.clf()
    sns.heatmap(np.abs(self.density), annot=True, vmin=0, vmax=1, cbar=True, cmap='Blues', alpha=0.9)
    if title: plt.suptitle(title)
    plt.tight_layout()
    plt.show()

  def plots(self, title='|phi>'):
    if os.environ.get('IGNORE_PLOTS'): return

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
      other = get_I(self.n_qubits)
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
    if other is Meta.Null: return self
    assert isinstance(other, Gate), f'other should be a Gate, but got {type(other)}'
    assert self.n_qubits == other.n_qubits, f'qubit count mismatch {self.n_qubits} != {other.n_qubits}'
    return Gate(self.v @ other.v)

  def __rmul__(self, other: Meta.Null) -> Gate:
    assert other is Meta.Null, f'other should be Meta.Null, but got {type(other)}'
    return self.__mul__(other)

  def __lshift__(self, other: Gate) -> Gate:
    '''
      Grammar sugar of **inplace** u = (gates * other * some) * u, nice to build a circuit module :)
        u = some << other << gates
      is eqv to
        u = (gates * other * some) * u
    '''
    assert isinstance(other, Gate), f'other should be a Gate, but got {type(other)}'
    self.v = (other * self).v
    return self

  def __matmul__(self, other: Union[Gate, int]) -> Gate:
    '''
      H @ X: tensor product of two quantum gate
      H @ 3 = H @ H @ H, tensor product by self n_times
    '''
    return super().__matmul__(other)

  def __rmatmul__(self, other: Meta.Null) -> Gate:
    return super().__rmatmul__(other)

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


''' Gate '''

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
U = lambda theta, phi, lmbd: Gate([                                                     # universal Z-Y decomposition
  [          cos(theta/2), -(e^(i* lmbd)     *sin(theta/2))],
  [e^(i*phi)*sin(theta/2),   e^(i*(lmbd+phi))*cos(theta/2)],
])
U1 = lambda alpha, beta, gamma, delta: Ph(alpha) * RZ(beta) * RY(gamma) * RZ(delta)     # universal Z-Y decomposition with global phase
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
CNOTNOT = (I @ SWAP) * (CNOT @ I) * (I @ SWAP) * (CNOT @ I)

Is = { }                      # identity gate caching
def get_I(n:int) -> Gate:
  if n not in Is: Is[n] = I @ n
  return Is[n]

def Control(u: Gate) -> Gate:
  '''
    Two-qubits CU (controlled unitary) gate, the control qubit is the higher one:
      [ I O
        O U ]
    where I is 1-qubit identity, U is 1-qubits unitary, Os are zeros
  '''
  assert u.n_qubits == 1, 'the controllee should be single-qubit gate'
  n_qubits = 2
  v = np.eye(2**n_qubits, dtype=u.v.dtype)
  v[-2:, -2:] = u.v
  return Gate(v)


''' State '''

def v(string: str) -> State:
  assert string, 'string should not be empty'
  qubits = { '0': v0, '1': v1 }
  v = qubits[string[0]]
  for c in string[1:]:
    v = v @ qubits[c]
  return v

v0 = State.zero()   # |0>
v1 = State.one()    # |1>
h0 = H | v0         # |+>
h1 = H | v1         # |->
bell_state = CNOT * (H @ I) | State.zero(2)   # |00>+|11>
ghz_state = (I @ CNOT) * (CNOT @ I) * (H @ I @ I) | State.zero(3)   # |000>+|111>


''' Measure '''

Measure = lambda n=1000: (lambda: n)
M0 = MeasureOp([
  [1, 0],
  [0, 0],
])
M1 = MeasureOp([
  [0, 0],
  [0, 1],
])


''' Algorithm / Circuit '''

def sSWAP(n_qubits=3) -> Gate:
  '''
    sSWAP: n-qubits skipping SWAP, which only swap the first qubit with the last
      ---x---  ---x---
      ---|---  ---|---
      ---x---  ---|---
               ---x---
    shows sSWAP(3) swapping |ijk> -> |kji> and sSWAP(4) swapping |abcd> -> |dbca>
  '''
  assert isinstance(n_qubits, int) and n_qubits >= 2, f'n_qubits should be an integer >=2 but got {n_qubits}'

  is_odd   = n_qubits % 2 == 1
  n_bubble = n_qubits // 2 - 1

  # prepare bubble swaps
  swaps: List[Gate] = [
    get_I(j) @ SWAP @ get_I(n_qubits - 2*(j+2)) @ SWAP @ get_I(j)
      for j in range(n_bubble)
  ]

  # bubble swap
  u: Gate = None
  for swap in swaps: u = swap * u

  # core swap
  if is_odd:
    '''
      ---x-----x---   i -> j -> j -> k
      ---x--x--x---   j -> i -> k -> j 
      ------x------   k -> k -> i -> i
    '''
    mid_swap1 = get_I(n_bubble) @ SWAP @ get_I(n_bubble + 1)
    mid_swap2 = get_I(n_bubble + 1) @ SWAP @ get_I(n_bubble)
    u = (mid_swap1 * mid_swap2 * mid_swap1) * u
  else:
    '''
      ---x---   i -> j
      ---x---   j -> i
    '''
    mid_swap = get_I(n_bubble) @ SWAP @ get_I(n_bubble)
    u = mid_swap * u

  # bubble swap (inverse)
  for swap in reversed(swaps): u = swap * u
  return u

def QFT(n_qubits=2, run_circuit=True) -> Gate:
  '''
    Linear basis transform alike DFT: 
      - https://en.wikipedia.org/wiki/Quantum_Fourier_transform
      - https://zhuanlan.zhihu.com/p/474941485
      - https://zhuanlan.zhihu.com/p/361711215
      - https://blog.csdn.net/qq_43270444/article/details/118607318
    Usage like:
      - encode cstate binary string to the phase (exponent factor) of qubits, so that
        - a single qubit classic state (eg. |0>) will be decomposed to a series Σi wi|i> of basis |i>, where weight vector wi is periodic in phase
        - a superposition state (eg. a|110>+b|011>) will be decomposed to a series ΣjΣi wji|i> of basis |i>, where weight matrix wij is periodic in phase along both axis
    The formula:
      |j> = (Σk e^(2*pi*i*(j*k/N))|k>) / sqrt(N), where N=2**k
    The unitary:
      [  1    1       1     ...    1
         1    w      w^2    ...  w^(N-1)
         1   w^2     w^4    ... w^2(N-1)
         ...
         1 w^(N-1) w^2(N-1) ... w^(N-1)^2 ]
    where w = e^(2*pi*i/N) is N=2^n equal devision of the circumference
  '''
  assert isinstance(n_qubits, int) and n_qubits >= 1, f'n_qubits should be an integer >=1 but got {n_qubits}'

  if run_circuit:
    '''
      Rk = P(2*pi/2^k), 2^k equal devision of the circumference
        [ 1        0
          0  e^(2*pi*i/2^k) ]
    '''

    n = n_qubits  # N is the phase angle unit (kind of FT resolution)
    CRk = { k: Control(P(2*pi/2**k)) for k in range(2, n+1) }   # CR2 ~ CRn
    # caching to reuse gates
    sSWAPs = { }
    def get_sSWAP(n:int) -> Gate:
      if n not in sSWAPs: sSWAPs[n] = sSWAP(n)
      return sSWAPs[n]

    u: Gate = None
    # apply H-CRk set
    for j in range(1, n+1):       # process each qubit |j>, from high |1> to low |n>
      # Hadamard gate on |j>
      u = (get_I(j-1) @ H @ get_I(n-j)) * u

      # CRx gates
      for k in range(2, n-(j-1)+1):   # qubit |j> need apply from CRx[2] to CRx[n-(j-1)]
        # prepare sswap |j+1> <-> |j+k-1>
        if k > 2:
          sswap = get_I(j) @ get_sSWAP(k-1) @ get_I(n-j-k+1)
        else:
          sswap = Meta.Null

        # apply sswap move |j+k-1> -> |j+1>
        u = sswap * u
        # apply CRk on |j,j+1>
        u = (get_I(j-1) @ CRk[k] @ get_I(n-j-1)) * u
        # apply sswap move |j+1> -> |j+k-1> (inverse)
        u = sswap * u

    # apply final sSwap set
    for j in range(n // 2):
      u = (get_I(j) @ get_sSWAP(n-2*j) @ get_I(j)) * u
    return u

  else:   # a cheaty way that constructs the unitary directly :(
    N = 2**n_qubits
    u = np.empty([N, N], dtype=DTYPE)
    w = e^(2*pi*i/N)
    for j in range(N):
      for k in range(N):
        u[j, k] = w**(j*k)
    u /= np.sqrt(N)
    return Gate(u)

iQFT = lambda n_qubits=2, run_circuit=True: QFT(n_qubits, run_circuit).dagger

def phase_estimate(u:Gate, phi:State=None, n_prec:int=4) -> State:
  '''
    Estimate the eigen value of a unitary U with eigen vector |phi>, i.e.:
      - https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm
      - https://zhuanlan.zhihu.com/p/84568388
      - https://blog.csdn.net/qq_45777142/article/details/109904362
    The formula:
      U|phi> = e^(2*pi*i*theta)|phi>
    note that eigen value is a global phase, it can be further reduced to the phase angle `theta`
    NOTE: 总之，相位估计可以给定一个特征向量的情况下，估计一个酉算子的一个对应特征值的相位。
  '''
  assert isinstance(n_prec, int) and n_prec >= 1, 'n_prec should be an integer >=1'

  t = n_prec
  # when U's eigen vector is also unknown, just uniformly random init and take it as 
  # kinda superposition of eigen vectors, we still have non-zero probability to get it :pray:
  if phi is None: phi = H | v('0' * u.n_qubits)

  '''
    |0>--H------------------------x--------|0>+e^2*pi*i(2^(t-1)*theta)|1>--|      |             (prec: 0.00..01, 1/2^(t-1))
            (repeat t times)     ...                                       | iQFT |--|theta>
    |0>--H--------------x---------|--------|0>+e^2*pi*i(2^1*theta)|1>------|      |             (prec: 0.1, 1/2=0.5)
    |0>--H-----x--------|---------|--------|0>+e^2*pi*i(2^0*theta)|1>------|      |             (prec: 1)
    |u>-----|U^2^0|--|U^2^1|--|U^2^(t-1)|--|u>    (aka. |phi> kept unchanged)
  '''
  # apply H set
  c = (H @ t) @ I     # t+1 qubits
  # apply C-U series
  for j in range(t):
    # prepare sSwap |t-j> <-> |t>
    sswap = get_I(t-j-1) @ (sSWAP(j+1) if j > 0 else I) @ I
    # apply sSwap
    c = sswap * c
    # apply control-U
    c = get_I(t-1) @ Control(u ^ (2**j))    # first register |00..0> controls on second register |phi>
    # inverse sSwap
    c = sswap * c
  # apply iQFT
  c = (iQFT(t) @ I) * c

  return c | (v('0' * t) @ phi)

def amplitude_encode(b: np.ndarray) -> State:
  ''' Amplitude encoding a unit vector b to |b> '''
  assert (np.linalg.norm(b) - 1.0) < 1e-5, 'b should be a unit vector'

  theta = 2 * np.arccos(b[0])
  return (Z if b[1] < 0 else I) * RY(theta) | v0

def HHL(A: np.ndarray, b: np.ndarray, t0=2*pi, r=4) -> State:
  '''
    Solve linear equations in a quantum manner:
      - https://arxiv.org/abs/1110.2232
      - https://en.wikipedia.org/wiki/Quantum_algorithm_for_linear_systems_of_equations
    Implementation of the toy HHL circuit solving a minimal 2x2 system using only 4 qubits
    given in essay "Quantum Circuit Design for Solving Linear Systems of Equations" by Yudong Cao, et al.
      - https://arxiv.org/abs/0811.3171

    q0: |0>──────────────────────────────────────────────┤RY(pi/8)├┤RY(pi/16)├─────────
    q1: |0>─┤H├───────────────────■───────X──────■───┤H├X─────■────────┼─────┼        ┼
    q2: |0>─┤H├────■──────────────┼───────X┤H├┤S.dag├───X──────────────■─────|U.dagger┼
    q3: |b>─┤exp(iA(t0/4))├┤exp(iA(t0/2))├───────────────────────────────────┼        ┼
  '''
  import scipy.linalg as spl

  assert np.allclose(A, A.conj().T), 'A should be a hermitian'
  assert (np.linalg.norm(b) - 1.0) < 1e-5, 'b should be a unit vector'

  ''' enc |b> '''
  enc_b = v('000') @ amplitude_encode(b)

  ''' QPE '''
  u = I @ H @ H @ I

  u_A1 = spl.expm(1j* A * (t0/4))
  u << (get_I(2) @ Control(Gate(u_A1)))

  swap12 = I @ SWAP @ I
  u << swap12
  u_A2 = spl.expm(1j* A * (t0/2))
  u << (get_I(2) @ Control(Gate(u_A2)))
  u << swap12

  u << swap12
  u << (get_I(2) @ H @ I)
  u << (I @ Control(S.dagger) @ I)
  u << (I @ H @ get_I(2))
  u << swap12

  QPE = u

  ''' RY '''
  swap01 = SWAP @ get_I(2)
  u =  swap01 * (Control(RY(2*pi/2**r)) @ get_I(2)) * swap01
  u << swap12
  u << swap01 * (Control(RY(pi/2**r)) @ get_I(2)) * swap01
  u << swap12

  CR = u

  ''' iQPE '''
  iQPE = QPE.dagger

  ''' final state '''
  return QPE << CR << iQPE | enc_b


if __name__ == '__main__':
  from code import interact
  interact(local=globals())
