#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/15 

from tiny_q import *

# Bell-state: quantum teleport
# https://en.wikipedia.org/wiki/Bell_state

bell_state = CNOT * (H @ I) | State.zero(2)
bell_state.info()

print('results:', bell_state > Measure())

bell_state.plots('bell state')


# GHZ state
# https://en.wikipedia.org/wiki/Greenberger%E2%80%93Horne%E2%80%93Zeilinger_state

ghz_state = (I @ CNOT) * (CNOT @ I) * (H @ I @ I) | State.zero(3)
ghz_state.info()

print('results:', ghz_state > Measure())

ghz_state.plots('ghz state')
