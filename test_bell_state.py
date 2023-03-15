#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/15 

from tiny_q import *

# Bell-state: quantum teleport
# https://en.wikipedia.org/wiki/Bell_state

bell_state: State = CNOT * (H @ I) | State.zero(2)

print('state:', bell_state)
print('results:', bell_state.measure())

bell_state.plot_density('bell state')
