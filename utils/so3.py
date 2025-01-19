import os
import numpy as np
import torch
import math
import pdb
from scipy.spatial.transform import Rotation


MIN_EPS, MAX_EPS, N_EPS = 0.0005, 4, 2000
# MIN_EPS, MAX_EPS, N_EPS = 0.0005, 4, 1000 
X_N, L = 2000, 2000
# X_N, L = 1000, 1000

"""
    Preprocessing for the SO(3) sampling and score computations, truncated infinite
    series are computed and then cached to memory, therefore the precomputation is
    only run the first time the repository is run on a machine
"""

omegas = np.linspace(0, np.pi, X_N + 1)[1:]

# _omegas_array = np.load('.so3_omegas_array4.npy')
# _cdf_vals = np.load('.so3_cdf_vals4.npy')
# _score_norms = np.load('.so3_score_norms4.npy')
# _exp_score_norms = np.load('.so3_exp_score_norms4.npy')

print("9")

def _compose(r1, r2):  # R1 @ R2 but for Euler vecs
    return Rotation.from_matrix(Rotation.from_rotvec(r1).as_matrix() @ Rotation.from_rotvec(r2).as_matrix()).as_rotvec()


def _expansion(omega, eps, L=2000):  # the summation term only
    l_vec = np.arange(L).reshape(-1, 1)
    p = (
            (2 * l_vec + 1) 
            * np.exp(-l_vec * (l_vec + 1) * eps ** 2 / 2)
            * np.sin(omega * (l_vec + 1 / 2)) 
            / np.sin(omega / 2)
            ).sum(0)
    return p

def expansion_vals(omega, eps, L=2000):
    """
    omega: X_N
    eps: N_EPS
    returns: N_EPS, X_N
    """
    x_n, = omega.shape
    n_eps, = eps.shape

    l_vec = torch.arange(L, device=omega.device)[None,:] # 1,L
    omega = omega[:,None] # X_N,1
    t1 = (2 * l_vec + 1) * torch.exp(-l_vec * (l_vec + 1) * eps[:,None] ** 2 / 2) # N_EPS,L
    t2 = torch.sin(omega * (l_vec + 1 / 2)) # X_N,L
    t3 = torch.sin(omega / 2) # X_N,1
    assert t1.shape == (n_eps,L), f"{t1.shape=} != {n_eps,L}"
    assert t2.shape == (x_n,L)
    assert t3.shape == (x_n,1)
    ans = torch.zeros((n_eps, x_n), dtype=omega.dtype, device=omega.device)
    chunk = 10
    ranges = list(range(0, L, chunk)) + [L]
    # for l in range(L):
    for b, e in zip(ranges[:-1], ranges[1:]):
        ans += (t1[:,None,b:e] * t2[None,:,b:e]).sum(dim=2) / t3[None,:,0]
    return ans # N_EPS, X_N 


def _density(expansion, omega, marginal=True):  # if marginal, density over [0, pi], else over SO(3)
    if marginal:
        return expansion * (1 - np.cos(omega)) / np.pi
    else:
        return expansion / 8 / np.pi ** 2  # the constant factor doesn't affect any actual calculations though

def density_vals(expansion, omega, marginal=True):
    """
    expansion: N_EPS, X_N
    omega: X_N 
    returns: N_EPS, X_N
    """
    if marginal:
        return expansion * (1 - torch.cos(omega)) / torch.pi
    else:
        return expansion / 8 / np.pi ** 2

def _score(exp, omega, eps, L=2000):  # score of density over SO(3)
    l_vec = np.arange(L).reshape(-1, 1)  # L,1 
    hi = np.sin((l_vec + 1 / 2) * omega) # L,X_N
    dhi = (l_vec + 1 / 2) * np.cos((l_vec + 1 / 2) * omega) # L,X_N
    lo = np.sin(omega / 2) # X_N
    dlo = 1 / 2 * np.cos(omega / 2) # X_N
    dSigma = ((2 * l_vec + 1) * np.exp(-l_vec * (l_vec + 1) * 
                                       eps**2 / 2) * (lo * dhi - hi * dlo) / lo ** 2).sum(0)
    # return dSigma / exp
    return dSigma 

def score_vals(expansion, omega, eps, L=2000):
    """
    expansion: N_EPS, X_N
    omega: X_N
    eps: N_EPS
    returns: N_EPS, X_N
    """
    assert expansion.dtype == torch.float64
    assert omega.dtype == torch.float64
    assert eps.dtype == torch.float64

    l_vec = torch.arange(L, dtype=omega.dtype, device=omega.device)[None,:] # 1,L 
    omega = omega[:,None] # X_N,1
    hi = torch.sin((l_vec + 1 / 2) * omega) # X_N,L
    dhi = (l_vec + 1 / 2) * torch.cos((l_vec + 1 / 2) * omega) # X_N,L
    lo = torch.sin(omega / 2) # X_N,1
    dlo = 1 / 2 * torch.cos(omega / 2) # X_N,1
    t1 = (2 * l_vec + 1) * torch.exp(-l_vec * (l_vec + 1) * eps[:,None]**2 / 2) # N_EPS,L
    t2 = (lo * dhi - hi * dlo) / lo ** 2 # X_N,L
    dsigma = torch.zeros((N_EPS, X_N), dtype=omega.dtype, device=omega.device)
    for l in range(L):
        dsigma += t1[:,None,l] * t2[None,:,l]
    # return dsigma / expansion # N_EPS, X_N
    return dsigma 

_eps_array = 10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), N_EPS)
_omegas_array = np.linspace(0, np.pi, X_N + 1)[1:]

"""
_exp_vals = np.asarray([_expansion(_omegas_array, eps, L=L) for eps in _eps_array])

_pdf_vals = np.asarray([_density(_exp, _omegas_array, marginal=True) for _exp in _exp_vals])

_cdf_vals = np.asarray([_pdf.cumsum() / X_N * np.pi for _pdf in _pdf_vals])

_score_norms = np.asarray([_score(_exp_vals[i], _omegas_array, _eps_array[i], L=L) 
                           for i in range(len(_eps_array))])

_exp_score_norms = np.sqrt(np.sum(_score_norms**2 * _pdf_vals, axis=1) 
                           / np.sum(_pdf_vals, axis=1) / np.pi)
# print("11")
"""

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
"""
NOTE: must directly initialize _eps_ten and _omegas_ten rather than use 
torch.linspace.  Slight differences between torch.linspace and np.linspace magnify
and create different values in score_vals
"""

_eps_ten = torch.tensor(_eps_array, device=dev)
_omegas_ten = torch.tensor(_omegas_array, device=dev)
_exp_vals_ten = expansion_vals(_omegas_ten, _eps_ten, L=L)
_pdf_vals_ten = density_vals(_exp_vals_ten, _omegas_ten, marginal=True)
_cdf_vals_ten = torch.cumsum(_pdf_vals_ten, dim=1) / X_N * torch.pi
_score_norms_ten = score_vals(_exp_vals_ten, _omegas_ten, _eps_ten, L=L)
_exp_score_norms_ten = torch.sqrt(torch.sum(_score_norms_ten**2 * _pdf_vals_ten, axis=1) 
                                  / torch.sum(_pdf_vals_ten, axis=1) / torch.pi)

def check_equal(msg, ary, ten, rtol=1e-7, atol=1e-8):
    if isinstance(ten, torch.Tensor):
        ten = ten.cpu().numpy()
    if isinstance(ary, torch.Tensor):
        ary = ary.cpu().numpy()

    assert ary.shape == ten.shape, f"{msg} shape mismatch"
    far = np.abs(ary - ten) > atol + rtol * np.abs(ten)
    num_far = far.sum().item()
    print(f"{msg}: {num_far} far out of {ten.size}")

"""
check_equal('eps', _eps_array, _eps_ten, rtol=0.0, atol=0.0)
check_equal('omegas', _omegas_array, _omegas_ten, rtol=0.0, atol=0.0)
check_equal('exp', _exp_vals, _exp_vals_ten)
check_equal('pdf', _pdf_vals, _pdf_vals_ten)
check_equal('cdf', _cdf_vals, _cdf_vals_ten)
check_equal('score_norms', _score_norms, _score_norms_ten)
check_equal('exp_score_norms', _exp_score_norms, _exp_score_norms_ten)

eps: 0 far out of 2000
omegas: 0 far out of 2000
exp: 0 far out of 4000000
pdf: 0 far out of 4000000
cdf: 0 far out of 4000000
score_norms: 2518 far out of 4000000
exp_score_norms: 0 far out of 2000

"""

# kludge - functions below expect numpy arrays
_eps_array = _eps_ten.cpu().numpy()
_omegas_array = _omegas_ten.cpu().numpy()
_exp_vals = _exp_vals_ten.cpu().numpy()
_pdf_vals = _pdf_vals_ten.cpu().numpy()
_cdf_vals = _cdf_vals_ten.cpu().numpy()
_score_norms = _score_norms_ten.cpu().numpy()
_exp_score_norms = _exp_score_norms_ten.cpu().numpy()


def sample(eps):
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    x = np.random.rand()
    return np.interp(x, _cdf_vals[eps_idx], _omegas_array)


def sample_vec(eps):
    x = np.random.randn(3)
    x /= np.linalg.norm(x)
    return x * sample(eps)


def score_vec(eps, vec):
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    om = np.linalg.norm(vec)
    return np.interp(om, _omegas_array, _score_norms[eps_idx]) * vec / om


def score_norm(eps):
    eps = eps.numpy()
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS-1)
    return torch.from_numpy(_exp_score_norms[eps_idx]).float()
