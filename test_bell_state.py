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


# GHZ state
# https://en.wikipedia.org/wiki/Greenberger%E2%80%93Horne%E2%80%93Zeilinger_state

ghz_state = (I @ CNOT) * (CNOT @ I) * (H @ I @ I) | State.zero(3)

print('state:', ghz_state)
print('results:', ghz_state.measure())

ghz_state.plot_density('ghz state')
