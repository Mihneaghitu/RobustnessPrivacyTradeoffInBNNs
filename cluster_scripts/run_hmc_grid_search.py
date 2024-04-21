import sys

sys.path.append('../')
sys.path.append('../probabilistic/HMC/')

from probabilistic.HMC.grid_search import run_dp_sweep, run_no_dp_sweep
from probabilistic.HMC.grid_search_adv import run_adv_no_dp_sweep

#* Command line format would be:
#! python run_hmc_grid_search.py [STD/ADV] [(dp/no-dp - for STD) (FGSM/PGD/IBP) - for ADV]

def grid_search_privacy_types(arg: str):
    match arg:
        case 'dp':
            run_dp_sweep()
        case 'no-dp':
            run_no_dp_sweep()
        case _:
            run_dp_sweep()
            run_no_dp_sweep()

def grid_search_attack_types(arg: str):
    match arg:
        case 'FGSM':
            run_adv_no_dp_sweep('FGSM')
        case 'IBP':
            run_adv_no_dp_sweep('IBP')
        case _:
            run_adv_no_dp_sweep('PGD')

# case 1 no adv:
match sys.argv[1]:
    case 'STD':
        grid_search_privacy_types(sys.argv[2])
    case 'ADV':
        grid_search_attack_types(sys.argv[2])
