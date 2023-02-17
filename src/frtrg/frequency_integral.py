# Copyright 2021 Valentin Bruch <valentin.bruch@rwth-aachen.de>
# License: MIT
"""
Kondo FRTRG, module for exact solution of a frequency integral
"""

import numpy as np
from .rtrg import RGfunction, array_shift
from .reservoirmatrix import ReservoirMatrix

def integral_taylor_n(a, b, n):
    """
    Taylor expansion to order n in (a-b) of
        ∞      1     1
        ∫ dω ————— —————
        0    ω + a ω + b
    Only odd orders contribute, such that for integers m:
    n=2*m and n=2*m-1 yield the same result.
    """
    x = (a-b)/(a+b)
    result = 1
    for m in range(3, n+1, 2):
        result += x**(m-1) / m
    result *= 2/(a+b)
    return result

def integral_baseline(a, b):
    """
    Frequency integral
        ∞      1     1
        ∫ dω ————— —————
        0    ω + a ω + b
    in the approximation used by default in the RG equations
    """
    return 0.5*(1/a + 1/b)

def integral_exact(a, b):
    """
    Exact result of the frequency integral
        ∞      1     1
        ∫ dω ————— —————
        0    ω + a ω + b
    """
    a, b = np.broadcast_arrays(a, b)
    result = np.log(a/b)/(a-b)
    fix_results = np.isclose(a, b, rtol=1e-4, atol=1e-15)
    result[fix_results] = 2/(a+b)[fix_results]
    return result

def integral(a, b, method):
    """
    Solve the integral
        ∞      1     1
        ∫ dω ————— —————
        0    ω + a ω + b
    method > 0: return Taylor series of order <method>
    method == -1: return exact solution
    method == -2: return baseline approximation for comparison to
                  default RG equations
    """
    if method > 0:
        return integral_taylor_n(a, b, method)
    # Here the match statement of python 3.10 would be better, but the code
    # should also run on old systems (like some computer clusters).
    elif method == -1:
        return integral_exact(a, b)
    elif method == -2:
        return integral_baseline(a, b)
    else:
        raise ValueError(f"Invalid method: {method}")

def matrix_integral(a, b, c, method):
    """
    Solve
        ∞      1       1
        ∫ dω ————— b —————
        0    ω + a   ω + c
    where a, b, c are numpy arrays representing matrices.
    """
    a_eigvals, a_reigvecs = np.linalg.eig(a)
    a_leigvecs = np.linalg.inv(a_reigvecs)
    if c is a:
        c_eigvals = a_eigvals
        c_reigvecs = a_reigvecs
        c_leigvecs = a_leigvecs
    else:
        c_eigvals, c_reigvecs = np.linalg.eig(c)
        c_leigvecs = np.linalg.inv(c_reigvecs)
    return matrix_integral_diagonalized(a_eigvals, a_reigvecs, a_leigvecs, b, c_eigvals, c_reigvecs, c_leigvecs, method)

def integral_eigenval_arrays(a_eigvals, c_eigvals, method):
    return integral(
            a_eigvals.reshape((*a_eigvals.shape, 1)),
            c_eigvals.reshape((*c_eigvals.shape[:-1],1,c_eigvals.shape[-1])),
            method)

def matrix_integral_diagonalized(a_eigvals, a_reigvecs, a_leigvecs, b, c_eigvals, c_reigvecs, c_leigvecs, method):
    """
    Solve
        ∞      1       1
        ∫ dω ————— b —————
        0    ω + a   ω + c
    where a, b, c are matrices.
    For a and c the eigenvectors and eigenvalues are given.
    """
    matrix = a_leigvecs @ b @ c_reigvecs
    matrix *= integral_eigenval_arrays(a_eigvals, c_eigvals, method)
    return a_reigvecs @ matrix @ c_leigvecs

def floquet_matrix_integral(a:RGfunction, b:RGfunction, c:RGfunction, method:int):
    """
    Solve
        ∞      1       1
        ∫ dω ————— b —————
        0    ω + a   ω + c
    where a, b, c are Floquet matrices.
    """
    assert a.global_properties is b.global_properties is c.global_properties
    b_shifted = b.shift_energies(a.voltage_shifts)
    c_shifted = c.shift_energies(a.voltage_shifts + b.voltage_shifts)
    a_eigvals, a_reigvecs, a_leigvecs = a.diagonalize()
    if a is c_shifted:
        try:
            cached_integral = a.cached_integral
        except AttributeError:
            cached_integral = integral_eigenval_arrays(a_eigvals, a_eigvals, method)
            a.cached_integral = cached_integral
        matrix = a_reigvecs @ ((a_leigvecs @ b_shifted.values @ a_reigvecs) * cached_integral) @ a_leigvecs
    else:
        c_eigvals, c_reigvecs, c_leigvecs = c_shifted.diagonalize()
        matrix = matrix_integral_diagonalized(a_eigvals, a_reigvecs, a_leigvecs, b_shifted.values, c_eigvals, c_reigvecs, c_leigvecs, method)
    return RGfunction(a.global_properties, matrix, symmetry=0)

def reservoir_matrix_integral_reference(a:RGfunction, b:ReservoirMatrix, c:RGfunction, method:int):
    """
    Slow reference implementation. Solve
        ∞      1       1
        ∫ dω ————— b —————
        0    ω + a   ω + c
    where a and c are Floquet matrices and b is a reservoir matrix.
    """
    assert a.global_properties is b.global_properties is c.global_properties
    result = ReservoirMatrix(a.global_properties, symmetry=0)
    for ij in ((0,0), (0,1), (1,0), (1,1)):
        result[ij] = floquet_matrix_integral(a, b[ij], c, method)
    return result

def reservoir_matrix_integral(a:RGfunction, b:ReservoirMatrix, c:RGfunction, method:int):
    """
    Solve
        ∞      1       1
        ∫ dω ————— b —————
        0    ω + a   ω + c
    where a and c are Floquet matrices and b is a reservoir matrix.
    """
    assert a.global_properties is b.global_properties is c.global_properties
    c_eigvals, c_reigvecs, c_leigvecs = c.diagonalize()
    c_shift = a.voltage_shifts + b.voltage_shifts
    if c_shift != 0:
        c_shifted = c.shift_energies(c_shift)
        if not hasattr(c_shifted, "eigvals"):
            c_shifted.eigvals = array_shift(c_eigvals, c_shift)
            c_shifted.reigvecs = array_shift(c_reigvecs, c_shift)
            c_shifted.leigvecs = array_shift(c_leigvecs, c_shift)
    if c_shift != -1:
        c_plus = c.shift_energies(c_shift + 1)
        if not hasattr(c_plus, "eigvals"):
            c_plus.eigvals = array_shift(c_eigvals, c_shift + 1)
            c_plus.reigvecs = array_shift(c_reigvecs, c_shift + 1)
            c_plus.leigvecs = array_shift(c_leigvecs, c_shift + 1)
    if c_shift != 1:
        c_minus = c.shift_energies(c_shift - 1)
        if not hasattr(c_minus, "eigvals"):
            c_minus.eigvals = array_shift(c_eigvals, c_shift - 1)
            c_minus.reigvecs = array_shift(c_reigvecs, c_shift - 1)
            c_minus.leigvecs = array_shift(c_leigvecs, c_shift - 1)
    result = ReservoirMatrix(a.global_properties, symmetry=0)
    for ij in ((0,0), (0,1), (1,0), (1,1)):
        result[ij] = floquet_matrix_integral(a, b[ij], c, method)
    return result

def calculate_chi(gamma, z):
    """
    Calculate χ from Γ and Z:
    TODO: check sign of μ?

        χ = Z (E + μ + NΩ + iΓ)
    """
    assert gamma.global_properties is z.global_properties
    inv_pi = 1j*gamma
    vb = z.voltage_branches
    nmax = z.nmax
    omega = z.omega
    energy = z.energy
    if inv_pi._values.ndim == 3:
        for v in range(-vb, vb+1):
            inv_pi._values[(v+vb, *np.diag_indices(2*nmax+1))] += \
                    energy \
                    + v*z.vdc \
                    + omega*np.arange(-nmax, nmax+1)
    elif inv_pi._values.ndim == 2:
        inv_pi._values[np.diag_indices(2*nmax+1)] += energy + omega*np.arange(-nmax, nmax+1)
    else:
        raise ValueError("invalid dimension of gamma")
    if z.mu is not None:
        inv_pi += z.mu
    return z @ inv_pi
