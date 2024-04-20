@ECHO OFF

SETLOCAL
SET IGNORE_PLOTS=true

PUSHD %~dp0examples

python test_basics.py
python test_basics_sanity.py
python test_basics_plot.py

python test_coin.py
python test_bell_state.py

python test_Deutsch_Jozsa.py
python test_phase_estimate.py
python test_QFT.py

POPD
