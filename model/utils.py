import numpy as np


def Beta_Function(x, alpha, beta):
    """Beta function"""
    from scipy.special import gamma
    return gamma(alpha + beta) / gamma(alpha) / gamma(beta) * x ** (alpha - 1) * (1 - x) ** (beta - 1)

def build_anneal_beta(beta_init = 0, beta_end = 0.3, max_epoch = 200):
    beta_init = beta_init
    init_length = int(max_epoch / 4)
    anneal_length = int(max_epoch / 4)
    beta_inter = Beta_Function(np.linspace(0,1,anneal_length),1,4)
    beta_inter = beta_inter / 4 * (beta_init - beta_end) + beta_end
    beta_list = np.concatenate([np.ones(init_length) * beta_init, beta_inter, 
                                    np.ones(max_epoch - init_length - anneal_length + 1) * beta_end])
    
    return beta_list