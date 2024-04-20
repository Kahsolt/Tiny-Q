#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/18 

from tiny_q import *
import numpy as np

# show what does RX, RY, RZ gate actually does, to understand VQC
#   RX: move magnitude between |0> and |1>
#   RY: move magnitude between real and imag part
#   RZ: alike RY, but relative phase on |1>
#
#                   RX                       RY                      RZ
#-------------------------------------------------------------------------------------------
#  -pi:   [ 0.5+0.5j, -0.5+0.5j]   [ 0.5-0.5j, -0.5-0.5j]  [-0.5+0.5j, -0.5-0.5j]
# -pi/2:  [ 0.7+0.7j,  0.0+0.0j]   [ 0.7+0.0j,  0.0-0.7j]  [ 0.0+0.7j,  0.0-0.7j]
#   0:    [ 0.5+0.5j,  0.5-0.5j]   [ 0.5+0.5j,  0.5-0.5j]  [ 0.5+0.5j,  0.5-0.5j]
# +pi/2:  [ 0.0+0.0j,  0.7-0.7j]   [ 0.0+0.7j,  0.7+0.0j]  [ 0.7+0.0j,  0.7+0.0j]
#  +pi:   [-0.5-0.5j,  0.5-0.5j]   [-0.5+0.5j,  0.5+0.5j]  [ 0.5-0.5j,  0.5+0.5j]
#-------------------------------------------------------------------------------------------
# NOTE: 0.7 denotes 0.7071 = 1/sqrt(2)


q = V | v0
gates = {
  'RX': RX,
  'RY': RY,
  'RZ': RZ,
}

for name, gate in gates.items():
  print(f'{name}')
  for theta in np.linspace(-np.pi, np.pi, 17):
    print(f' {theta:+.3f}: {list((gate(theta) | q).v)}')
  print()
