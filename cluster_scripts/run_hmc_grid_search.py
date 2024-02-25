import sys

sys.path.append('../')
sys.path.append('../probabilistic/HMC/')

from probabilistic.HMC.grid_search import run_dp_sweep, run_no_dp_sweep

if sys.argv[1] == 'dp':
    run_dp_sweep()
elif sys.argv[1] == 'no-dp':
    run_no_dp_sweep()
else:
    run_dp_sweep()
    run_no_dp_sweep()
