import sys

sys.path.append('../')
sys.path.append('../probabilistic/HMC/')

from probabilistic.HMC.grid_search import run_dp_sweep, run_no_dp_sweep
from probabilistic.HMC.grid_search_adv import run_adv_no_dp_sweep

#* Command line format would be:
#! python run_hmc_grid_search.py [STD/ADV] [(dp/no-dp - for STD) (fgsm/pgd/ibp) - for ADV]

def grid_search_privacy_types(arg: str):
    match arg:
        case 'dp':
            run_dp_sweep()
        case 'no-dp':
            run_no_dp_sweep()
        case _:
            run_dp_sweep()
            run_no_dp_sweep()

# case 1 no adv:
match sys.argv[1]:
    case 'STD':
        grid_search_privacy_types(sys.argv[2])
    case 'ADV':
        run_adv_no_dp_sweep(sys.argv[2])
