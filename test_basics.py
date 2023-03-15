#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/15 

import matplotlib.pyplot as plt ; plt.ion()

from tiny_q import *


## use matmul operator '@' for system expansion (tensor product)

v0 = State.zero() ; print(v0)
v1 = State.one()  ; print(v1)

v00 = v0 @ v0 ; print(v00)
v01 = v0 @ v1 ; print(v01)
v10 = v1 @ v0 ; print(v10)
v11 = v1 @ v1 ; print(v11)

v0011 = v00 @ v11 ; print(v0011)
v011  = v01 @ v1  ; print(v011)

v000 = State.zero(3) ; print(v000)
v11  = State.one(2)  ; print(v11)
v00011 = v000 @ v11  ; print(v00011)
v0110  = v01  @ v10  ; print(v00011)

assert v000 @ v11 == v00 @ v011 == v0 @ v0011
assert v0011 @ v0 == v0 @ v0110


## use pipe operator '|' for gate application
r = H | v0
print(r)
r = X | v1
print(r)

# auto broadcast 
r = H | State.one(3)
print(r)


## Use mul operator '*' for gate composition
ops: Gate = RY(np.pi/3) * Z * H
r: State = ops | v0
print(r)
r.plot_density('RY(np.pi/3) * Z * H | v0')

ops: Gate = RY(-np.pi/3) * S
r: State = ops | v1
print(r)
r.plot_density('RY(-np.pi/3) * S | v1')
