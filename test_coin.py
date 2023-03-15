#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/15 

from tiny_q import *

# quantum coin tossing

q = H | State.zero(2)

print('state:', q)
print('is_pure:', q.is_pure)
print('amp:', q.amp)
print('prob:', q.prob)
print('density:', q.density)

r = q.measure()
print(r)
