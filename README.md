# Tiny-Q

    Tiny-Q is a minimal framework to show quantum computation basics in a tensor/matrix perspective view.

----

![demo](img/demo.png)


The main difference from existing framesworks is that, 
we distinguish operations by different python operator to make formula syntax more clear :)

```python
# use mul * for gate composition
# u = Gate * Gate
u = X * H

# use pipe | for gate application
# q = Gate | State
q = X | v0

# use matmul @ for system expansion (gate/state tensor dot)
#u = Gate @ Gate
u = H @ I
#q = State @ State
q = v0 @ v1

# use > for measurement
# r = State > Measure
r = H | v0 > Measure
r = CNOT * (H @ I) | State.zero(2) > Measure
```

⚪ API stubs

```python
class State:
  .n_qubits -> int
  .is_pure -> bool
  .amp -> np.ndarray            # amplitude
  .prob -> np.ndarray           # probobilty distribution
  .density -> np.ndarray        # density matrix
  .plot_density()
  .measure() -> Dict[str, int]  # batch measurement
  .__eq__() -> bool             # state equality (ignoring global phase)
  .__matmul__() -> State        # v0 @ v1, state expansion
  .__gt__() -> str              # v0 > Measure, single measurement

class Gate:
  .n_qubits -> int
  .is_unitary -> bool           # unitary (should always be True)
  .is_hermitian -> bool         # hermitian (True for most gates)
  .__pow__() -> Gate            # H**alpha, gate self-power
  .__mul__() -> Gate            # X * H: gate composition
  .__matmul__() -> Gate         # X @ H: gate expansion
  .__or__() -> State            # X | v0: gate application
```

----

by Armit
2023/03/15 
