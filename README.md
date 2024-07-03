# Tiny-Q

    Tiny-Q is a minimal framework to show quantum computation basics in a tensor/matrix perspective view.

----

![demo](img/demo.png)


### Install

⚪ From PyPI

ℹ Due to the name conflict with an [existing pypi package](https://pypi.org/project/TinyQ/), you have to address `TinyQSim` rather than `TinyQ` :(

- `pip install TinyQSim`

⚪ From source

- `git clone https://github.com/Kahsolt/Tiny-Q`
- `cd Tiny-Q`
- `pip install .`
- you can see [exmaples](/examples/) or run `run_unnittest.cmd` to verify installation


### Features

⚪ Tiny-Q interactive playground

Start interactive python shell with Tiny-Q environment: `python -m tiny_q`

```python
(InteractiveConsole)
>>> v('00')
array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dtype=complex64)
>>> H
array([[ 0.7071+0.j,  0.7071+0.j],
       [ 0.7071+0.j, -0.7071+0.j]], dtype=complex64)
>>> CNOT
array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]], dtype=complex64)
>>> q = CNOT * (H @ I) | v('00')
>>> q.info()
|phi>
  state: [0.7071+0.j 0.    +0.j 0.    +0.j 0.7071+0.j]
  amp: [0.7071 0.     0.     0.7071]
  prob: [0.5 0.  0.  0.5]
  density: [[0.5+0.j 0. +0.j 0. +0.j 0.5+0.j]
 [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
 [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
 [0.5+0.j 0. +0.j 0. +0.j 0.5+0.j]]
  trace: (0.99999994+0j)

>>> q > Measure()
{'00': 489, '01': 0, '10': 0, '11': 511}
>>> q < Measure
>>> q.info()
|phi>
  state: [1.+0.j 0.+0.j 0.+0.j 0.+0.j]
  amp: [1. 0. 0. 0.]
  prob: [1. 0. 0. 0.]
  density: [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
 [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
 [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
 [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
  trace: (1+0j)

>>> Ctrl+D
now exiting InteractiveConsole...
```

⚪ Tiny-Q syntax / notations

Tiny-Q strictly distinguishes the following four categories of operations, an re-assigin each to a different python operator, making expresssions syntax formula-like thus more clear :)

- `system expansion (@)` is the hadamard product
- `gate composition (*) and gate construction (<<)` are the matrix product
- `gate application (|)` is the quantum state evolution
- `virtual measure (>) and real measure (<)` are the quantum measurements

```python
# use matmul @ for system expansion (gate/state tensor product)
# u = Gate @ Gate
u = H @ I
# q = State @ State
q = v0 @ v1
# NOTE: the low qubit starts from **right**
q = v_high @ v_mid @ v_low    # => |high,mid,low>

# use mul * for gate composition
# u = Gate * Gate (more like math formula, reducing from **right**)
u = X * H
u = gate3 * gate2 * gate1
# use lshift << for gate construction
# u = Gate << Gate (more like programs, running from **left**, modify **inplace**)
u = H << X
u = gate1 << gate2 << gate3

# use pipe | for gate application
# q = Gate | State
q = X | v0
# use pipe | for pauli expectation or state fidelity
# E = State | Gate | State
E = State.rand() | (X @ Z) | State.rand()
# fid = State | State
fid = (X | v('01')) | (H | v('01'))

# use > for virtual measure, the state does not really change
# r = State > Measure, single shot
r = H | v0 > Measure
# r = State > Measure(count), multi shots
r = CNOT * (H @ I) | State.zero(2) > Measure(1000)
# p = State > State, project by state
p = v0 > h0
# p = State > MeasureOp, project by measure operator
p = h0 > M0

# use < for real measure, the state will collapse
# State < Measure
q = CNOT * (H @ I) | v('00')
q < Measure
```

### API stub

```python
class Meta:
  .n_qubits -> int          # qubit count of current system
  .dagger -> Meta           # dagger of State/Gate/MeasureOp

class State(Meta):
  .zero() -> State          # alloc a |0> string
  .one() -> State           # alloc a |1> string
  .rand() -> State          # get a random state (usually as test stub)
  .__eq__() -> bool         # state equality (ignoring global phase)
  .__matmul__() -> State    # v0 @ v1: state expansion
  .__lt__() -> Union        # v0 < Measure, real measure with state collapse
  .__gt__() -> Union        # v0 > Measure|Measure()|State|MeasureOp, virtual measurements
  .is_pure -> bool          # purity
  .amp -> ndarray           # amplitude
  .prob -> ndarray          # probabilty distribution
  .density -> ndarray       # density matrix
  .trace -> float           # trace of density matrix
  .info()                   # quick show info
  .plot_prob()              # plot probabilty distribution
  .plot_density()           # plot density matrix
  .plots()                  # plot all figures

class Gate(Meta):
  .__eq__() -> bool         # gate equality
  .__neg__() -> Gate        # -H, global negative
  .__xor__() -> Gate        # H^alpha, gate self-power
  .__mul__() -> Gate        # X * H: gate composition
  .__lshift__() -> Gate     # H << X: gate composition (reverse order of __mul__)
  .__matmul__() -> Gate     # X @ H: gate expansion
  .__or__() -> State        # X | v0: gate application
  .is_unitary -> bool       # unitary (should always be True)
  .is_hermitian -> bool     # hermitian (True for most gates)
  .info()                   # quick show info

class MeasureOp(Meta):
  .check_completeness() -> bool
```

----

by Armit
2023/03/15 
