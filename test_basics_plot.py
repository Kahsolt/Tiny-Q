#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/15 

from tiny_q import *

ops = RY(pi/3) * Z * H
q = ops | v0
q.info('RY(pi/3)ZH|0>')
q.plots('RY(pi/3) * Z * H | v0')

ops = RY(-pi/3) * S
q = ops | v1
q.info('RY(-pi/3)S|1>')
q.plots('RY(-pi/3) * S | v1')


# more complex circuit
g0 = RY(-pi/3) * S
g1 = Z * T * X
g2 = RY(pi/7) * H
g3 = iSWAP * (g1 @ g2)
g4 = (CNOT @ I)* (g0 @ g3)
cirq = Toffoli * g4
cirq.info()

state = State.zero(3)
state.info()

q = cirq | state
q.info()
q.plots()
