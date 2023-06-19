from src.comparators.compare_functions.cka import cka
from src.comparators.compare_functions.cca import cca
from src.comparators.compare_functions.l2 import l2
from src.comparators.compare_functions.ps_inv import ps_inv
from src.comparators.compare_functions.correlation import correlation
from src.comparators.compare_functions.sq_sum import sq_sum
from src.comparators.compare_functions.lr import lr
from src.comparators.compare_functions.tsne import tsne
from src.comparators.compare_functions.ls_orth import ls_orth
from src.comparators.compare_functions.r2 import r2
from src.comparators.compare_functions.ls_sum import ls_sum

from src.comparators.compare_functions.cka_torch import cka as cka_torch
from src.comparators.compare_functions.lr_torch import lr as lr_torch

def get_comparator_function(str_comparator):
    str_comparator = str_comparator.lower()
    dispatcher = {
        'cka' : cka,
        'cca' : cca,
        'l2' : l2,
        'ps_inv' : ps_inv,
        'ls_orth' : ls_orth,
        'corr' : correlation,
        'sq_sum' : sq_sum,
        'lr' : lr,
        'r2' : r2,
        'ls_sum' : ls_sum,
        'lr_torch' : lr_torch,
        'cka_torch' : cka_torch,
    }
    
    if str_comparator not in dispatcher:
        raise ValueError('{} is unknown comparator.'.format(str_comparator))

    return dispatcher[str_comparator]
