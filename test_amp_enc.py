#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/07 

from tiny_q import *

# Amplitude encoding
# - https://qml.baidu.com/tutorials/machine-learning/encoding-classical-data-into-quantum-states.html


err = 0.0
for i in range(1000):
  b = np.random.uniform(size=[2], low=-1.0, high=1.0)
  b /= np.linalg.norm(b)

  q = amplitude_encode(b)
  
  if i < 10:
    print('b:', b)
    print('|b>:', q)
  
  err += (np.abs(b - q.v.real)).sum()

print()
print('error:', err)
