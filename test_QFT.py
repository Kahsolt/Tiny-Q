#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/15 

from tiny_q import *

# quantum fourier transform

# test sanity of components
assert (sSWAP(2) | v('10'))  .cbit == '01'
assert (sSWAP(2) | v('01'))  .cbit == '10'
assert (sSWAP(3) | v('100')) .cbit == '001'
assert (sSWAP(3) | v('001')) .cbit == '100'
assert (sSWAP(4) | v('1000')).cbit == '0001'
assert (sSWAP(4) | v('1100')).cbit == '0101'
assert (sSWAP(4) | v('0001')).cbit == '1000'
assert (sSWAP(4) | v('0011')).cbit == '1010'

for k in range(2, 5+1):
  print(f'R{k}:')
  print(P(2*pi/2**k))


# test sanity
for n in range(1, 5+1):
  qft_t = QFT(n, run_circuit=False)
  qft_c = QFT(n, run_circuit=True)
  try: assert qft_t == qft_c
  except: print(f'mismatch when n = {n}')


# what happens on bell_state?
q = bell_state
q.info('bell_state')
q.plots('bell_state')
q_qft = QFT(q.n_qubits) | q
q_qft.info('bell_state QFT')
q_qft.plots('bell_state QFT')

q = ghz_state
q.info('ghz_state')
q.plots('ghz_state')
q_qft = QFT(q.n_qubits) | q
q_qft.info('ghz_state QFT')
q_qft.plots('ghz_state QFT')


# show what it does
# => just basis transform from |01> to |+-> ??
for n in range(1, 3+1):
  qft = QFT(n)

  # |00..0>
  q = v('0' * n)
  q.info('q')
  q.plots('q')
  q_qft = qft | q
  q_qft.info('q QFT')
  q_qft.plots('q QFT')

  # |++..+> = H|00..0>
  p = H | q
  p.info('p')
  p.plots('p')
  p_qft = qft | p
  p_qft.info('p QFT')
  p_qft.plots('p QFT')
