#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/15 

from tiny_q import *

# quantum fourier transform
# https://en.wikipedia.org/wiki/Quantum_Fourier_transform

# sanity check
for n in range(2, 10+1):
  sswap = sSWAP(n)
  for val in range(n):
    sbit = bin(val)[2:].rjust(n, '0')
    rbit = sbit[-1] + sbit[1:-1] + sbit[0]
    assert (sswap | v(sbit)).cbit == rbit

for n in range(1, 7+1):
  qft_t = QFT(n, run_circuit=False)
  qft_c = QFT(n, run_circuit=True)
  assert qft_t == qft_c

# QFT & iQFT
q = bell_state
q.info()
q_qft = QFT(q.n_qubits) | q
q_qft.info()
q_iqft = iQFT(q.n_qubits) | q_qft
q_iqft.info()


# what happens on bell_state & ghz_state?
q = bell_state
q.info ('bell_state')
q.plots('bell_state')

qft = QFT(q.n_qubits)
q_qft = q
for n in range(1, 10+1):
  q_qft = qft | q_qft
  q_qft.info (f'QFT @ {n} | bell_state')
  q_qft.plots(f'QFT @ {n} | bell_state')

q = ghz_state
q.info ('ghz_state')
q.plots('ghz_state')

qft = QFT(q.n_qubits)
q_qft = q
for n in range(1, 10+1):
  q_qft = qft | q_qft
  q_qft.info (f'QFT @ {n} | ghz_state')
  q_qft.plots(f'QFT @ {n} | ghz_state')


# show what QFT it actually does...
# => just basis transform from |01> to |+-> ??
for sbit in ['0', '10', '11', '101']:
  q = QFT(len(sbit)) | v(sbit)
  q.info (f'QFT | {sbit}>')
  q.plots(f'QFT | {sbit}>')
