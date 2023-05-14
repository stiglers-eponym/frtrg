# Copyright 2022 Valentin Bruch <valentin.bruch@rwth-aachen.de>
# License: MIT
# type: ignore
"""
Kondo FRTRG, main module for RG calculations

Floquet real-time renormalization group implementation for the
spin 1/2 isotropic Kondo model.

Example usage:
>>> import numpy as np
>>> from frtrg.kondo import Kondo
>>> nmax = 10
>>> vb = 3
>>> # Compute the RG flow in 2 different ways
>>> kondo1 = Kondo(
...     unitary_transformation=True,
...     omega=10,
...     nmax=nmax,
...     padding=8,
...     vdc=4,
...     vac=5,
...     voltage_branches=vb)
>>> kondo2 = Kondo(
...     unitary_transformation=False,
...     omega=10,
...     nmax=nmax,
...     padding=0,
...     vdc=4,
...     vac=5,
...     voltage_branches=vb)
>>> solver1 = kondo1.run()
>>> solver2 = kondo2.run()
>>> # Check if the results agree
>>> np.abs(kondo1.gammaL[:,nmax] - kondo2.gammaL[:,nmax]).max()
2.4691660784226606e-05
>>> np.abs(kondo1.deltaGammaL[:,nmax] - kondo2.deltaGammaL[:,nmax]).max()
1.2758585339583961e-05
>>> np.abs(kondo1.gamma[vb,:,nmax] - kondo2.gamma[vb,:,nmax]).max()
0.00018744930313729924
>>> np.abs(kondo1.z[vb,:,nmax] - kondo2.z[vb,:,nmax]).max()
1.421144031910071e-05

Further information:
https://arxiv.org/abs/2206.06263 and https://vbruch.eu/frtrg.html
"""

import hashlib
import numpy as np
from time import time, process_time
from scipy.special import jn as bessel_jn
from scipy.fftpack import fft
from scipy.integrate import solve_ivp
from scipy.optimize import newton
from . import settings
from .rtrg import GlobalRGproperties, RGfunction
from .reservoirmatrix import ReservoirMatrix, \
        einsum_34_12_43, \
        einsum_34_12_43_double, \
        product_combinations
from .compact_rtrg import SymRGfunction
from . import frequency_integral

# Log times (global variables)
REF_TIME = time()
LAST_LOG_TIME = REF_TIME


def driving_voltage(tau, *fourier_coef):
    """
    Generate function of time given Fourier coefficients.

    tau = t/T so that 0 <= tau <= 1.
    fourier_coef[n] = V_{n+1}/Ω

    The result is given in the same units as fourier_coef.
    """
    res = np.zeros_like(tau)
    for n, c in enumerate(fourier_coef, 1):
        res += (c * np.exp(2j*np.pi*n*tau)).real
    return 2*res


def driving_voltage_integral(tau, *fourier_coef):
    """
    Compute time-integral given Fourier coefficients.

    tau = t/T so that 0 <= tau <= 1.
    fourier_coef[n] = V_{n+1}/Ω

              t
    return  Ω ∫dt' V(t') ,  t = tau*T
              0

    fourier_coef should be in units of Ω, then the result has
    unit hbar/e = 1
    """
    res = np.zeros_like(tau)
    for n, c in enumerate(fourier_coef, 1):
        res += (c/n * (np.exp(2j*np.pi*n*tau) - 1)).imag
    return 2*res


def gen_init_matrix(nmax, *fourier_coef, resonant_dc_shift=0, resolution=5000):
    """
    Generate Floquet matrix of the bare coupling without scalar
    coupling prefactor.

    fourier_coef must be in units of Ω:
    fourier_coef[n] = V_{n+1}/Ω
    TODO: signs have been chosen such that the result looks correct
    """
    if len(fourier_coef) == 1 or np.allclose(fourier_coef[1:], 0):
        # Simple driving, only one frequency:
        coef = bessel_jn(
                np.arange(-2*nmax+resonant_dc_shift, 2*nmax+resonant_dc_shift+1),
                -2*fourier_coef[0])
    elif len(fourier_coef) == 0:
        coef = np.zeros(4*nmax+1)
        coef[2*nmax-resonant_dc_shift] = 1
    else:
        # Anharmonic driving: use FFT
        assert resolution > 2*nmax + abs(resonant_dc_shift)
        assert resolution % 2 == 0
        fft_coef = fft(np.exp(
                    -1j*driving_voltage_integral(
                        np.linspace(0, 1, resolution, endpoint=False),
                        *fourier_coef
                    )
                ))
        coef = np.ndarray(4*nmax+1, dtype=np.complex128)
        coef[2*nmax-resonant_dc_shift:] = fft_coef[:2*nmax+resonant_dc_shift+1].conjugate() / resolution
        coef[:2*nmax-resonant_dc_shift] = fft_coef[-2*nmax+resonant_dc_shift:].conjugate() / resolution
    init_matrix = np.ndarray((2*nmax+1, 2*nmax+1), dtype=np.complex128)
    for j in range(2*nmax+1):
        init_matrix[:,j] = coef[2*nmax-j:4*nmax+1-j]
    return init_matrix


def solveTV0_scalar_order2(
        d,
        tk = 1, # TODO: prior to version 14.15 this was 0.32633176486110027
        rtol = 1e-8,
        atol = 1e-10,
        full_output = False,
        **solveopts):
    """
    Solve the ODE for second order truncated RG equations in
    equilibrium at T=V=0 for scalars from 0 to d.
    returns: (gamma, z, j, solver)

    The default value of tk is adjusted to yield comparible results to
    the case of 3rd order truncation for d=1e9, rtol=1e-8, atol=1e-10,
    voltage_branches=4
    """
    # Initial conditions
    j0 = 2/(np.pi*3**.5)
    z0 = 1/(2*j0+1)
    theta0 = 1/z0 - 1
    gamma0 = tk * np.exp(1/(2*j0)) / j0

    def ode_function_imaxis(lmbd, values):
        """RG eq. for Kondo model on imaginary axis, ODE of functions of variable lmbd = Λ"""
        gamma, theta, j = values
        dgamma = theta
        dtheta = -4*j**2/(lmbd + gamma)
        dj = dtheta/2
        return np.array([dgamma, dtheta, dj])

    result = solve_ivp(
            ode_function_imaxis,
            (0, d),
            np.array([gamma0, theta0, j0]),
            t_eval = None if full_output else (d,),
            rtol = rtol,
            atol = atol,
            **solveopts)
    assert result.success

    gamma, theta, j = result.y[:, -1]
    z = 1/(1+theta)
    return (gamma, z, j, result)


def solveTV0_scalar(
        d,
        tk = 1,
        rtol = 1e-8,
        atol = 1e-10,
        full_output = False,
        **solveopts):
    """
    Solve the ODE in equilibrium at T=V=0 for scalars from 0 to d.
    returns: (gamma, z, j, solver)
    """
    # Initial conditions
    jbar = 2/(np.pi*np.sqrt(3))
    j0 = jbar/(1-jbar)**2
    # Θ := d/dΛ Γ = 1/Z - 1
    theta0 = (1-jbar)**-2 - 1
    gamma0 = np.sqrt((1-jbar)/jbar)*np.exp(1/(2*jbar)) * tk

    # Solve on imaginary axis

    def ode_function_imaxis(lmbd, values):
        """RG eq. for Kondo model on imaginary axis, ODE of functions of variable lmbd = Λ"""
        gamma, theta, j = values
        dgamma = theta
        dtheta = -4*j**2/(lmbd + gamma)
        dj = dtheta*(1 + j/(1 + theta))/2
        return np.array([dgamma, dtheta, dj])

    t_eval = solveopts.pop("t_eval", None) if full_output else (d,)
    result = solve_ivp(
            ode_function_imaxis,
            (0, d),
            np.array([gamma0, theta0, j0]),
            t_eval = t_eval,
            rtol = rtol,
            atol = atol,
            **solveopts)
    assert result.success

    gamma, theta, j = result.y[:, -1]
    z = 1/(1+theta)
    return (gamma, z, j, result)


def solveTV0_Utransformed(
        d,
        properties,
        tk = 1,
        truncation_order = 3,
        rtol = 1e-8,
        atol = 1e-10,
        **solveopts):
    """
    Solve the ODE in equilibrium at T=V=0 to obtain initial conditions for Γ, Z and J.
    Here all time-dependence is assumed to be contained in J.

    returns: (gamma, z, j)

    Return values are arrays representing the quantities at energies shifted
    by Ω and μ as required for the initial conditions.
    If properties.resonant_dc_shift ≠ 0, then a larger array of energies
    is considered, equivalent to mapping nmax → nmax+resonant_dc_shift in
    the shape of properties.energies.
    """
    # Check input
    assert isinstance(properties, GlobalRGproperties)
    assert truncation_order in (2, 3)

    # Solve equilibrium RG equations from 0 to d.
    solveopts.update(rtol=rtol, atol=atol)
    if truncation_order == 3:
        gamma, z, j, scalar_solver = solveTV0_scalar(d, **solveopts)
    elif truncation_order == 2:
        gamma, z, j, scalar_solver = solveTV0_scalar_order2(d, **solveopts)

    # Solve for constant imaginary part, go to required points in complex plane.
    nmax = properties.nmax
    vb = properties.voltage_branches

    def ode_function_imconst(rE, values):
        """RG eq. for Kondo model for constant Im(E), ODE of functions of rE = Re(E)"""
        gamma, theta, j = values
        dgamma = theta
        dtheta = -4*j**2/(d - 1j*rE + gamma)
        if truncation_order == 3:
            dj = dtheta*(1 + j/(1 + theta))/2
        elif truncation_order == 2:
            dj = dtheta/2
        return -1j*np.array([dgamma, dtheta, dj])

    # Define flattened array of real parts of energy values, for which we want
    # to know, Γ, Z, J
    nmax_plus = nmax + abs(properties.resonant_dc_shift)
    if vb:
        energies_orig = \
                properties.energy.real \
                + properties.vdc*np.arange(-vb, vb+1).reshape((2*vb+1, 1)) \
                + properties.omega*np.arange(-nmax_plus, nmax_plus+1).reshape((1, 2*nmax_plus+1))
    else:
        energies_orig = properties.energy.real \
                + properties.omega * np.arange(-nmax_plus, nmax_plus+1)

    energies = energies_orig.flatten()
    energies_unique, inverse_indices = np.unique(energies, return_inverse=True)
    if energies_unique.size == 1:
        y = scalar_solver.y[:,-1].reshape((3,1))
    else:
        split_idx = np.searchsorted(energies_unique, 0)
        energies_left = energies_unique[:split_idx]
        energies_right = energies_unique[split_idx:]

        result_left = solve_ivp(
                ode_function_imconst,
                t_span = (0, energies_left[0]),
                y0 = np.array(scalar_solver.y[:,-1], dtype=np.complex128),
                t_eval = energies_left[::-1],
                **solveopts
            )
        assert result_left.success
        result_right = solve_ivp(
                ode_function_imconst,
                t_span = (0, energies_right[-1]),
                y0 = np.array(scalar_solver.y[:,-1], dtype=np.complex128),
                t_eval = energies_right,
                **solveopts
            )
        assert result_right.success
        y = np.concatenate((result_left.y[:,::-1], result_right.y), axis=1)
    gamma = y[0][inverse_indices].reshape(energies_orig.shape)
    z = ( 1/(1 + y[1]) )[inverse_indices].reshape(energies_orig.shape)
    j = y[2][inverse_indices].reshape(energies_orig.shape)

    return gamma, z, j


def solveTV0_untransformed(
        d,
        properties,
        tk = 1,
        truncation_order = 3,
        rtol = 1e-8,
        atol = 1e-10,
        **solveopts):
    """
    Solve the ODE in equilibrium at T=V=0 to obtain initial conditions for Γ, Z and J.
    Here all time-dependence is assumed to be included in the Floquet
    matrix μ used for the voltage shift.

    returns: (gamma, z, j, zj_square)

    Return values represent Floquet matrices of the quantities at energies
    shifted by μ as required for the initial conditions.
    """
    # Check input
    assert isinstance(properties, GlobalRGproperties)
    assert truncation_order in (2, 3)

    # Solve equilibrium RG equations from 0 to d.
    solveopts.update(rtol=rtol, atol=atol)
    if truncation_order == 3:
        gamma, z, j, scalar_solver = solveTV0_scalar(d, **solveopts)
    elif truncation_order == 2:
        gamma, z, j, scalar_solver = solveTV0_scalar_order2(d, **solveopts)

    # Solve for constant imaginary part, go to required points in complex plane.
    nmax = properties.nmax
    vb = properties.voltage_branches

    def ode_function_imconst(rE, values):
        'RG eq. for Kondo model for constant Im(E), ODE of functions of rE = Re(E)'
        gamma, theta, j = values
        dgamma = theta
        dtheta = -4*j**2/(d - 1j*rE + gamma)
        if truncation_order == 3:
            dj = dtheta*(1 + j/(1 + theta))/2
        elif truncation_order == 2:
            dj = dtheta/2
        return -1j*np.array([dgamma, dtheta, dj])

    shifts = properties.mu.values \
            + properties.omega*np.diag(np.arange(-nmax, nmax+1)).reshape((1,2*nmax+1,2*nmax+1))
    #assert np.allclose(shifts.imag, 0)
    eigvals, eigvecs = np.linalg.eigh(shifts)
    assert np.allclose(eigvals.imag, 0)
    assert all(np.allclose(eigvecs[i] @ np.diag(eigvals[i]) @ eigvecs[i].T.conjugate(), shifts[i]) for i in range(2*vb+1))

    energies = eigvals.flatten()
    energies_unique, inverse_indices = np.unique(energies, return_inverse=True)
    if energies_unique.size == 1:
        y = scalar_solver.y[:,-1].reshape((3,1))
    else:
        split_idx = np.searchsorted(energies_unique, 0)
        energies_left = energies_unique[:split_idx]
        energies_right = energies_unique[split_idx:]

        result_left = solve_ivp(
                ode_function_imconst,
                t_span = (0, energies_left[0]),
                y0 = np.array(scalar_solver.y[:,-1], dtype=np.complex128),
                t_eval = energies_left[::-1],
                **solveopts
            )
        assert result_left.success
        result_right = solve_ivp(
                ode_function_imconst,
                t_span = (0, energies_right[-1]),
                y0 = np.array(scalar_solver.y[:,-1], dtype=np.complex128),
                t_eval = energies_right,
                **solveopts
            )
        assert result_right.success
        y = np.concatenate((result_left.y[:,::-1], result_right.y), axis=1)
    gamma_raw = y[0][inverse_indices].reshape(eigvals.shape)
    z_raw = ( 1/(1 + y[1]) )[inverse_indices].reshape(eigvals.shape)
    j_raw = y[2][inverse_indices].reshape(eigvals.shape)

    gamma = np.einsum('kij,kj,klj->kil', eigvecs, gamma_raw, eigvecs.conjugate())
    z = np.einsum('kij,kj,klj->kil', eigvecs, z_raw, eigvecs.conjugate())
    j = np.einsum('kij,kj,klj->kil', eigvecs, j_raw, eigvecs.conjugate())
    if truncation_order == 3:
        zj_square = np.einsum('kij,kj,klj->kil', eigvecs, (j_raw*z_raw)**2, eigvecs.conjugate())
        return gamma, z, j, zj_square
    else:
        j_square = np.einsum('kij,kj,klj->kil', eigvecs, j_raw**2, eigvecs.conjugate())
        return gamma, z, j, j_square



class Kondo:
    """
    Kondo model with RG flow equations and routines for initial conditions.

    Always accessible properties:
        total_iterations: total number of calls to self.updateRGequations()
        global_properties: Properties for the RG functions, energy, ...
            Properties stored in global_properties can be accessed
            directly from self, e.g. using self.omega or self.energy.

    When setting initial values, the following properties are added:
        xL : asymmetry factor of the coupling, defaults to 0.5
        d : UV cutoff
        vac : AC voltage amplitude relative to Kondo temperature Tk
        ir_cutoff : IR cutoff, should be 0, but is adapted when RG flow
            is interrupted earlier
        z     : RGfunction, Z = 1/( 1 + i dΓ/dE )
        gamma : RGfunction, Γ  as in  Π = 1/(E+iΓ)
        deltaGammaL : RGfunction, δΓL (conductivity)
        deltaGamma : RGfunction, δΓ
        g2 : Reservoir matrix, coupling vertex G2
        g3 : Reservoir matrix, coupling vertex G3
        current : matrix in reservoir space fo RG functions,
            representing the current vertex I^L

    When running self.updateRGequations() the following properties are
    added:
        pi : RGfunction, Π = 1/(E+iΓ)
        zE : derivative of self.z with respect to E
        gammaE : derivative of self.gamma with respect to E
        deltaGammaE : derivative of self.deltaGamma with respect to E
        deltaGammaLE : derivative of self.deltaGammaL with respect to E
        g2E : derivative of self.g2 with respect to E
        g3E : derivative of self.g3 with respect to E
        currentE : derivative of self.current with respect to E
    """

    def __init__(self,
            unitary_transformation = True,
            nmax = 0,
            padding = 0,
            vdc = 0,
            vac = 0,
            omega = 0,
            d = 1e9,
            fourier_coef = None,
            voltage_branches = 0,
            include_Ga = False,
            solve_integral_exactly = False,
            integral_method = -15,
            resonant_dc_shift : 'DC bias voltages, multiples of Ω, positive int' = 0,
            xL : 'asymmetry factor' = 0.5,
            compact = 0,
            simplified_initial_conditions = False,
            improved_initial_conditions = True,
            truncation_order = 3,
            **rg_properties):
        """
        Create Kondo object, initialize global properties shared by all
        Floquet matrices.

        Expected arguments:

        omega : frequency Ω, in units of Tk
        nmax : size of Floquet matrix = (2*nmax+1, 2*nmax+1)
        padding : extrapolation to avoid Floquet matrix truncation
                effects, valid values: 0 ≤ padding ≤ 2*nmax-2
        vdc : DC voltage, in units of Tk, including voltage due to
                resonant_dc_shift.
        vac : AC voltage, in units of Tk
        voltage_branches : keep copies of Floquet matrices with energies
                shifted by n Vdc, n = ±1,...,±voltage_branches.
                Must be 0 or >=2
        resonant_dc_shift : Describe DC voltage vdc partially by shifts
                in the Floquet in the initial conditions.
                valid values: non-negative integers
        d : UV cutoff (convergence parameter)
        include_Ga : include vertex paramter Ga in RG equations
        solve_integral_exactly : diagonalize χ to solve integral in
                next-to-leading order terms exactly.
        integral_method : only relevant with solve_integral_exactly.
                positive integer: truncation order of Taylor series
                    representing the integral solution.
                -1: exact solution
                -2: baseline, should yield same results as
                    solve_integral_exactly == False
                -15: indicates that solve_integral_exactly == False
        xL = 1 - xR : asymmetry factor of coupling. Must fulfill 0 ≤ xL ≤ 1.
        clear_corners : improve convergence for large Floquet matrices
                by setting parts of the matrices to 0 after each
                multiplication. Handle with care!
                valid values: padding + clear_corners <= 2*nmax + 1
        compact : Use extra symmetry to improve efficiency for large
                matrices. compact != 0 requires unitary_transformation
                and the symmetry V(t+(π/Ω)) = - V(t). Valid values are:
                    0: don't use compact form.
                    1: compact form that ignores matrix elements which
                        are zero by symmetry
                    2: compact form which additionally uses symmetry of
                        the nonzero matrix elements. requires xL = 0.5.
        simplified_initial_conditions : use simplified initial
                conditions for the current kernel which lead to the same
                result in the limit of large D.
        improved_initial_conditions : use nonzero initial condition for
                derivative of current kernel Γ^L
        truncation_order : Truncation order of RG equations, must be 2 or 3.
        """
        self.global_properties = GlobalRGproperties(
                nmax = nmax,
                omega = omega,
                vdc = 0,
                mu = None,
                voltage_branches = voltage_branches,
                resonant_dc_shift = resonant_dc_shift,
                padding = padding,
                fourier_coef = fourier_coef,
                energy = 0j,
                **rg_properties)
        assert truncation_order in (2, 3)
        self.truncation_order = truncation_order
        self.global_settings = settings.export()
        self.unitary_transformation = unitary_transformation
        self.simplified_initial_conditions = simplified_initial_conditions
        self.improved_initial_conditions = improved_initial_conditions
        assert not (simplified_initial_conditions and improved_initial_conditions)
        self.include_Ga = include_Ga
        self.solve_integral_exactly = solve_integral_exactly
        if truncation_order == 2:
            assert not include_Ga
            assert not solve_integral_exactly
        if solve_integral_exactly:
            if padding > 0:
                settings.logger.warn("solve_integral_exactly and padding>0: will not handle padding consistenly")
            if integral_method == -15:
                self.integral_method = -1
                settings.logger.warn("setting integral_method = -1")
            else:
                assert integral_method > -3
                self.integral_method = integral_method
        else:
            if integral_method != -15:
                settings.logger.warn("parameter integral_method defined but unused!")
            self.integral_method = -15
        self.compact = compact
        self.vdc = vdc
        self.vac = vac
        self.xL = xL
        if xL != 0.5 or settings.IGNORE_SYMMETRIES:
            # Ga has the structure Ga[0,0] == - Ga[1,1].
            # This kind of symmetry is not implemented yet.
            # Workaround: set global_properties.symmetric = False and ga.symmetry = 0
            self.global_properties.symmetric = False
        if xL != 0.5:
            assert self.compact != 2

        # Some checks of the input
        if (vac or fourier_coef is not None) and self.nmax == 0:
            raise ValueError("Bad parameters: driving != 0 requires nmax > 0")
        if xL < 0 or xL > 1:
            raise ValueError("Bad parameter: need  0 <= xL <= 1")
        if resonant_dc_shift and abs(resonant_dc_shift) > self.nmax:
            raise ValueError("Bad parameters: resonant_dc_shift must be <= nmax")
        if compact or voltage_branches == 0:
            assert unitary_transformation
            assert self.vdc == omega*resonant_dc_shift
            if compact:
                assert voltage_branches == 0
                assert fourier_coef is None or np.allclose(fourier_coef[1::2], 0)
                # When resonant_dc_shift is finite, typically the driving does
                # not obey symmetry required for compact calculation.
                # This implementation cannot handle resonant_dc_shift in
                # combination with "compact" matrices.
                assert self.resonant_dc_shift == 0
        assert d.imag == 0.
        self.d = d

        if self.unitary_transformation:
            self.compact = compact
            self.global_properties.vdc = vdc - resonant_dc_shift*omega
        else:
            assert resonant_dc_shift == 0
            mu = np.zeros((2*nmax+1, 2*nmax+1), dtype=np.complex128)
            mu[np.diag_indices(2*nmax+1)] = vdc
            if fourier_coef is None:
                mu[np.arange(2*nmax), np.arange(1, 2*nmax+1)] = vac/2
                mu[np.arange(1, 2*nmax+1), np.arange(2*nmax)] = vac/2
            else:
                for i, f in enumerate(fourier_coef, 1):
                    mu[np.arange(2*nmax+1-i), np.arange(i, 2*nmax+1)] = f
                    mu[np.arange(i, 2*nmax+1), np.arange(2*nmax+1-i)] = f.conjugate()
            self.global_properties.mu = RGfunction(
                    self.global_properties,
                    np.arange(-voltage_branches, voltage_branches+1)\
                            .reshape((2*voltage_branches+1,1,1)) \
                            * mu.reshape((1,2*nmax+1,2*nmax+1)),
                    symmetry = -1)
        self.total_iterations = 0

    def __getattr__(self, name):
        """self.<name> is defined as shortcut for self.global_properties.<name>"""
        return getattr(self.global_properties, name)

    def getParameters(self):
        """
        Get most relevant parameters.
        The returned dict can be used to label this object.
        """
        return {
                'method' : 'J' if self.unitary_transformation else 'mu',
                'Ω' : self.omega,
                'nmax' : self.nmax,
                'padding' : self.padding,
                'Vdc' : self.vdc,
                'Vac' : self.vac,
                'V_branches' : self.voltage_branches,
                'Vac' : getattr(self, 'vac', None),
                'D' : getattr(self, 'd', None),
                'solveopts' : getattr(self, 'solveopts', None),
                'xL' : getattr(self, 'xL', None),
                }

    def initialize_untransformed(self,
            **solveopts : 'keyword arguments passed to solver',
            ):
        """
        Arguments:
        **solveopts: keyword arguments passed to the solver. Most relevant
                are rtol and atol.

        Get initial conditions for Γ, Z and G2 by numerically solving the
        equilibrium RG equations from E=0 to E=iD and for all required Re(E).
        Initialize G3, Iγ, δΓ, δΓγ.
        """
        sqrtxx = np.sqrt(self.xL*(1-self.xL))
        symmetry = 0 if settings.IGNORE_SYMMETRIES else 1


        #### Initial conditions from exact results at T=V=0
        # Get Γ, Z and J (G2) for T=V=0.
        # if truncation_order==2: zj0_square is just J^2
        gamma0, z0, j0, zj0_square = solveTV0_untransformed(
                d=self.d,
                properties=self.global_properties,
                truncation_order=self.truncation_order,
                **solveopts)

        # Create Γ and Z with the just calculated initial values.
        self.gamma = RGfunction(self.global_properties, gamma0, symmetry=symmetry)
        self.z = RGfunction(self.global_properties, z0, symmetry=symmetry)

        if self.include_Ga:
            # Create Ga:
            if self.global_properties.symmetric:
                self.ga_scalar = RGfunction(self.global_properties, np.zeros_like(j0), symmetry=-symmetry)
            else:
                self.ga = ReservoirMatrix(self.global_properties, symmetry=0)
                self.ga[0,0] = RGfunction(self.global_properties, np.zeros_like(j0), symmetry=-symmetry)
                self.ga[1,1] = RGfunction(self.global_properties, np.zeros_like(j0), symmetry=-symmetry)
                self.ga[0,1] = RGfunction(self.global_properties, np.zeros_like(j0), symmetry=0)
                self.ga[1,0] = RGfunction(self.global_properties, np.zeros_like(j0), symmetry=0)

        # Create G2 from J:    G2_{ij} = - 2 sqrt(x_i x_j) J
        self.g2 = ReservoirMatrix(self.global_properties, symmetry=symmetry)
        j_rgfunction = RGfunction(self.global_properties, j0, symmetry=symmetry)
        self.g2[0,0] = -2*self.xL * j_rgfunction
        self.g2[1,1] = -2*(1-self.xL) * j_rgfunction
        j_rgfunction.symmetry = 0
        self.g2[0,1] = -2*sqrtxx * j_rgfunction
        self.g2[1,0] = -2*sqrtxx * j_rgfunction


        ## Initial conditions for G3
        # G3 ~ Jtilde^2  with  Jtilde = Z J
        # Every entry of G3 will be of the following form (up to prefactors):
        self.g3 = ReservoirMatrix(self.global_properties, symmetry=-symmetry)
        g3_entry = RGfunction(
                self.global_properties,
                1j*np.pi * zj0_square,
                symmetry=-symmetry)
        self.g3[0,0] = 2*self.xL * g3_entry
        self.g3[1,1] = 2*(1-self.xL) * g3_entry
        g3_entry.symmetry = 0
        self.g3[0,1] = 2*sqrtxx * g3_entry
        self.g3[1,0] = 2*sqrtxx * g3_entry


        ## Initial conditions for current I^{γ=L} = J0 (1 - Jtilde)
        # Note that j0[self.voltage_branches] and z0[self.voltage_branches] are diagonal Floquet matrices.
        if self.truncation_order >= 3:
            current_entry = 2*sqrtxx * j0[self.voltage_branches] * ( \
                    np.identity(2*self.nmax+1, dtype=np.complex128) \
                    - j0[self.voltage_branches] * z0[self.voltage_branches] )
        else:
            current_entry = 2*sqrtxx * j0[self.voltage_branches]
        self.current = ReservoirMatrix(self.global_properties, symmetry=-symmetry)
        self.current[0,0] = RGfunction(
                self.global_properties,
                np.zeros_like(current_entry),
                symmetry=-symmetry)
        self.current[1,1] = RGfunction(
                self.global_properties,
                np.zeros_like(current_entry),
                symmetry=-symmetry)
        self.current[0,1] = RGfunction(
                self.global_properties,
                current_entry,
                symmetry=0)
        self.current[1,0] = RGfunction(
                self.global_properties,
                -current_entry,
                symmetry=0)

        ## Initial conditions for voltage-variation of Γ: δΓ
        self.deltaGamma = RGfunction(
                self.global_properties,
                np.zeros((3,2*self.nmax+1,2*self.nmax+1), dtype=np.complex128),
                symmetry = symmetry
                )

        ## Initial conditions for voltage-variation of current-Γ: δΓ_L
        # Note that zj0_square[self.voltage_branches] is a diagonal Floquet matrix.
        self.deltaGammaL = RGfunction(
                self.global_properties,
                3*np.pi*self.xL*(1-self.xL) * zj0_square[self.voltage_branches],
                symmetry = symmetry
                )


        ### Derivative of full current
        self.yL = RGfunction(
                self.global_properties,
                np.zeros((2*self.nmax+1,2*self.nmax+1), dtype=np.complex128),
                symmetry = -symmetry
                )
        mu_matrix = self.mu.reduced(shift=1)
        if self.improved_initial_conditions:
            gamma = gamma0[self.voltage_branches].diagonal()
            j = j0[self.voltage_branches].diagonal()
            z = z0[self.voltage_branches].diagonal()
            ddGammaL = 12j*np.pi*self.xL*(1-self.xL) * j**3/(self.d+gamma)
            if self.truncation_order >= 3:
                ddGammaL *= z**2*(1-z*j)
            self.yL = RGfunction(self.global_properties, mu_matrix.values * ddGammaL.reshape((1,-1)), symmetry=-symmetry)

        ### Full current, also includes AC current
        self.gammaL = mu_matrix @ self.deltaGammaL
        if self.simplified_initial_conditions:
            self.gammaL *= 0

        self.global_properties.energy = 1j*self.d

    def initialize_Utransformed(self,
            **solveopts : 'keyword arguments passed to solver',
            ):
        """
        Initialize all RG objects with unitary transformation.

        Arguments:
        **solveopts: keyword arguments passed to the solver. Most
                relevant are rtol and atol.

        Get initial conditions for Γ, Z and G2 by numerically solving
        the equilibrium RG equations from E=0 to E=iD and for all
        required Re(E). Then initialize G3, Iγ, δΓ, δΓγ, Yγ.
        """

        sqrtxx = np.sqrt(self.xL*(1-self.xL))
        symmetry = 0 if settings.IGNORE_SYMMETRIES else 1


        #### Initial conditions from exact results at T=V=0
        # Get Γ, Z and J (G2) for T=V=0.
        gamma0, z0, j0 = solveTV0_Utransformed(
                d = self.d,
                properties = self.global_properties,
                truncation_order = self.truncation_order,
                **solveopts)

        # Write T=V=0 results to Floquet index n=0.
        gammavalues = np.zeros(self.shape(), dtype=np.complex128)
        zvalues = np.zeros(self.shape(), dtype=np.complex128)
        jvalues = np.zeros(self.shape(), dtype=np.complex128)

        # construct diagonal matrices
        diag_idx = (..., *np.diag_indices(2*self.nmax+1))
        if self.resonant_dc_shift:
            gammavalues[diag_idx] = gamma0[...,self.resonant_dc_shift:-self.resonant_dc_shift]
            zvalues[diag_idx] = z0[...,self.resonant_dc_shift:-self.resonant_dc_shift]
            jvalues[diag_idx] = j0[...,self.resonant_dc_shift:-self.resonant_dc_shift]
        else:
            gammavalues[diag_idx] = gamma0
            zvalues[diag_idx] = z0
            jvalues[diag_idx] = j0

        zj0 = z0 * j0 if self.truncation_order >= 3 else j0
        zjvalues = zvalues * jvalues if self.truncation_order >= 3 else jvalues

        gamma0_red = gamma0[self.voltage_branches] if self.voltage_branches else gamma0
        z0_red = z0[self.voltage_branches] if self.voltage_branches else z0
        j0_red = j0[self.voltage_branches] if self.voltage_branches else j0
        zj0_red = zj0[self.voltage_branches] if self.voltage_branches else zj0

        RGclass = SymRGfunction if self.compact else RGfunction

        # Create Γ and Z with the just calculated initial values.
        if self.compact:
            self.gamma = SymRGfunction(self.global_properties, gammavalues, symmetry=symmetry, diag=True, offdiag=False)
            self.gamma.check_symmetry()
            self.z = SymRGfunction(self.global_properties, zvalues, symmetry=symmetry, diag=True, offdiag=False)
            self.z.check_symmetry()
        else:
            self.gamma = RGfunction(self.global_properties, gammavalues, symmetry=symmetry)
            self.gamma.check_symmetry()
            self.z = RGfunction(self.global_properties, zvalues, symmetry=symmetry)
            self.z.check_symmetry()

        if self.include_Ga:
            # Create Ga:
            if self.global_properties.symmetric:
                settings.logger.debug("Creating Ga scalar")
                if self.compact:
                    self.ga_scalar = SymRGfunction(self.global_properties, np.zeros_like(jvalues), symmetry=-symmetry, diag=True, offdiag=False)
                else:
                    self.ga_scalar = RGfunction(self.global_properties, np.zeros_like(jvalues), symmetry=-symmetry)
            else:
                settings.logger.debug("Creating Ga matrix")
                self.ga = ReservoirMatrix(self.global_properties, symmetry=symmetry)
                # TODO: check whether this works for RGclass with self.compact
                self.ga[0,0] = RGclass(self.global_properties, np.zeros_like(jvalues), symmetry=-symmetry)
                self.ga[1,1] = RGclass(self.global_properties, np.zeros_like(jvalues), symmetry=-symmetry)
                self.ga[0,1] = RGclass(self.global_properties, np.zeros_like(jvalues), symmetry=0)
                self.ga[1,0] = RGclass(self.global_properties, np.zeros_like(jvalues), symmetry=0)

        # Create G2 from J:    G2_{ij} = - 2 sqrt(x_i x_j) J
        self.g2 = ReservoirMatrix(self.global_properties, symmetry=symmetry)
        j_rgfunction = RGclass(self.global_properties, jvalues, symmetry=symmetry)
        self.g2[0,0] = -2*self.xL * j_rgfunction
        self.g2[1,1] = -2*(1-self.xL) * j_rgfunction
        if self.vac or self.resonant_dc_shift or self.fourier_coef is not None:
            # Coefficients are given by the Bessel function of the first kind.
            if self.fourier_coef is not None:
                init_matrix = gen_init_matrix(
                        self.nmax,
                        *(f/self.omega for f in self.fourier_coef),
                        resonant_dc_shift = self.resonant_dc_shift)
            else:
                init_matrix = gen_init_matrix(
                        self.nmax,
                        self.vac/(2*self.omega),
                        resonant_dc_shift = self.resonant_dc_shift)
            j_LR = np.einsum(
                    'ij,...j->...ij',
                    init_matrix,
                    j0[...,2*self.resonant_dc_shift:])
            j_RL = np.einsum(
                    'ji,...j->...ij',
                    init_matrix.conjugate(),
                    j0[...,:j0.shape[-1]-2*self.resonant_dc_shift])
            j_LR = RGfunction(self.global_properties, j_LR)
            j_RL = RGfunction(self.global_properties, j_RL)
            self.g2[0,1] = -2*sqrtxx * j_LR
            self.g2[1,0] = -2*sqrtxx * j_RL
        else:
            assert self.compact == 0
            j_rgfunction.symmetry = 0
            self.g2[0,1] = -2*sqrtxx * j_rgfunction
            self.g2[1,0] = -2*sqrtxx * j_rgfunction


        ## Initial conditions for G3
        # G3 ~ Jtilde^2  with  Jtilde = Z J
        # Every entry of G3 will be of the following form (up to prefactors):
        self.g3 = ReservoirMatrix(self.global_properties, symmetry=-symmetry)
        g3_entry = np.zeros(self.shape(), dtype=np.complex128)
        g3_entry[diag_idx] = 1j*np.pi * zjvalues[diag_idx]**2
        g3_entry = RGclass(self.global_properties, g3_entry, symmetry=-symmetry)
        self.g3[0,0] = 2*self.xL * g3_entry
        self.g3[1,1] = 2*(1-self.xL) * g3_entry
        if self.vac or self.resonant_dc_shift or self.fourier_coef is not None:
            g30 = 1j*np.pi*zj0**2
            g3_LR = np.einsum(
                    'ij,...j->...ij',
                    init_matrix,
                    g30[...,2*self.resonant_dc_shift:])
            g3_RL = np.einsum(
                    'ji,...j->...ij',
                    init_matrix.conjugate(),
                    g30[...,:g30.shape[-1]-2*self.resonant_dc_shift])
            g3_LR = RGfunction(self.global_properties, g3_LR)
            g3_RL = RGfunction(self.global_properties, g3_RL)
            self.g3[0,1] = 2*sqrtxx * g3_LR
            self.g3[1,0] = 2*sqrtxx * g3_RL
        else:
            assert self.compact == 0
            g3_entry.symmetry = 0
            self.g3[0,1] = 2*sqrtxx * g3_entry
            self.g3[1,0] = 2*sqrtxx * g3_entry


        ## Initial conditions for current I^{γ=L} = J0 (1 - Jtilde)
        if self.voltage_branches:
            if self.truncation_order >= 3:
                current_entry = np.diag( \
                        2*sqrtxx * jvalues[self.voltage_branches][diag_idx] \
                        * (1 - zjvalues[self.voltage_branches][diag_idx] ) )
            else:
                current_entry = np.diag(2*sqrtxx * jvalues[self.voltage_branches][diag_idx])
        else:
            if self.truncation_order >= 3:
                current_entry = np.diag(2*sqrtxx * jvalues[diag_idx] * (1 - zjvalues[diag_idx]))
            else:
                current_entry = np.diag(2*sqrtxx * jvalues[diag_idx])
        current_entry = RGclass(self.global_properties, current_entry, symmetry=-symmetry)
        self.current = ReservoirMatrix(self.global_properties, symmetry=-symmetry)
        if self.compact:
            self.current[0,0] = RGclass(self.global_properties, None, symmetry=symmetry)
            self.current[0,0].submatrix01 = np.zeros((self.nmax+1, self.nmax), np.complex128)
            self.current[0,0].submatrix10 = np.zeros((self.nmax, self.nmax+1), np.complex128)
        else:
            self.current[0,0] = 0*current_entry
        self.current[0,0].symmetry = -symmetry
        self.current[1,1] = self.current[0,0].copy()
        if self.vac or self.resonant_dc_shift or self.fourier_coef is not None:
            if self.truncation_order >= 3:
                i0 = j0_red * (1 - zj0_red)
            else:
                i0 = j0_red
            i_LR = np.einsum(
                    'ij,...j->...ij',
                    init_matrix,
                    i0[2*self.resonant_dc_shift:])
            i_RL = np.einsum(
                    'ji,...j->...ij',
                    init_matrix.conjugate(),
                    i0[:i0.size-2*self.resonant_dc_shift])
            i_LR = RGfunction(self.global_properties, i_LR)
            i_RL = RGfunction(self.global_properties, i_RL)
            self.current[0,1] =  2*sqrtxx * i_LR
            self.current[1,0] = -2*sqrtxx * i_RL
        else:
            assert self.compact == 0
            current_entry.symmetry = 0
            self.current[0,1] =  current_entry
            self.current[1,0] = -current_entry

        ## Initial conditions for voltage-variation of Γ: δΓ
        if self.compact:
            self.deltaGamma = SymRGfunction(
                    self.global_properties,
                    np.zeros(
                        (3,2*self.nmax+1,2*self.nmax+1) if self.voltage_branches else self.shape(),
                        dtype=np.complex128),
                    symmetry = symmetry,
                    diag = False,
                    offdiag = True,
                    )
        else:
            self.deltaGamma = RGfunction(
                    self.global_properties,
                    np.zeros(
                        (3,2*self.nmax+1,2*self.nmax+1) if self.voltage_branches else self.shape(),
                        dtype=np.complex128),
                    symmetry = symmetry
                    )

        ## Initial conditions for voltage-variation of current-Γ: δΓ_L
        self.deltaGammaL = RGclass(
                self.global_properties,
                None if self.compact else np.zeros((2*self.nmax+1, 2*self.nmax+1), dtype=np.complex128),
                symmetry = symmetry
                )
        if self.resonant_dc_shift:
            assert self.compact == 0
            self.deltaGammaL.values[diag_idx] = 3*np.pi*self.xL*(1-self.xL) \
                    * zj0_red[self.resonant_dc_shift:-self.resonant_dc_shift]**2
        else:
            diag_values = 3*np.pi*self.xL*(1-self.xL) * zj0_red**2
            if self.compact:
                assert diag_values.dtype == np.complex128
                self.deltaGammaL.submatrix00 = np.diag(diag_values[0::2])
                self.deltaGammaL.submatrix11 = np.diag(diag_values[1::2])
            else:
                self.deltaGammaL.values[diag_idx] = diag_values
            del diag_values


        ### Derivative of full current
        if self.compact:
            self.yL = SymRGfunction(
                    self.global_properties,
                    np.zeros((2*self.nmax+1,2*self.nmax+1), dtype=np.complex128),
                    symmetry=-symmetry,
                    diag=False,
                    offdiag=True,
                    )
        else:
            self.yL = RGfunction(
                    self.global_properties,
                    np.zeros((2*self.nmax+1,2*self.nmax+1), dtype=np.complex128),
                    symmetry=-symmetry
                    )

        ### Full current, also includes AC current
        if self.resonant_dc_shift == 0:
            # TODO: implement this with resonant shift
            mu = np.zeros((2*self.nmax+1, 2*self.nmax+1), dtype=np.complex128)
            mu[np.diag_indices(2*self.nmax+1)] = self.vdc
            if self.fourier_coef is None:
                mu[np.arange(2*self.nmax), np.arange(1, 2*self.nmax+1)] = self.vac/2
                mu[np.arange(1, 2*self.nmax+1), np.arange(2*self.nmax)] = self.vac/2
            else:
                for i, f in enumerate(self.fourier_coef, 1):
                    mu[np.arange(2*self.nmax+1-i), np.arange(i, 2*self.nmax+1)] = f
                    mu[np.arange(i, 2*self.nmax+1), np.arange(2*self.nmax+1-i)] = f.conjugate()
            mu = RGclass(self.global_properties, mu, symmetry=1)
            self.gammaL = mu @ self.deltaGammaL.reduced()
            if self.improved_initial_conditions:
                ddGammaL = 12j*np.pi*self.xL*(1-self.xL) * j0_red**3/(self.d+gamma0_red)
                if self.truncation_order >= 3:
                    ddGammaL *= z0_red**2*(1-z0_red*j0_red)
                if self.compact:
                    self.yL = SymRGfunction(self.global_properties, mu.values * ddGammaL.reshape((1,-1)), symmetry=-symmetry, diag=False, offdiag=True)
                else:
                    self.yL = RGfunction(self.global_properties, mu.values * ddGammaL.reshape((1,-1)), symmetry=-symmetry)
        else:
            self.gammaL = self.vdc * self.deltaGammaL.reduced()
            if self.improved_initial_conditions and self.vdc:
                if self.truncation_order >= 3:
                    self.yL.values += np.diag(self.vdc * 12j*np.pi*self.xL*(1-self.xL) * j0_red**3*z0_red**2*(1-z0_red*j0_red)/(self.d+gamma0_red))
                else:
                    self.yL.values += np.diag(self.vdc * 12j*np.pi*self.xL*(1-self.xL) * j0_red**3/(self.d+gamma0_red))
            if self.vac and self.fourier_coef is None:
                gammaL_AC = 3*np.pi*self.xL*(1-self.xL) * self.vac/2 * zj0_red**2
                if self.resonant_dc_shift:
                    gammaL_AC = gammaL_AC[...,self.resonant_dc_shift:-self.resonant_dc_shift]
                idx = (np.arange(1, 2*self.nmax+1), np.arange(2*self.nmax))
                self.gammaL.values[idx] = gammaL_AC[...,1:]
                idx = (np.arange(0, 2*self.nmax), np.arange(1, 2*self.nmax+1))
                self.gammaL.values[idx] = gammaL_AC[...,:-1]
                if self.improved_initial_conditions:
                    yL_AC = 12j*np.pi*self.xL*(1-self.xL) * self.vac/2 * j0_red**3/(self.d+gamma0_red)
                    if self.truncation_order >= 3:
                        yL_AC *= z0_red**2*(1-z0_red*j0_red)
                    if self.resonant_dc_shift:
                        yL_AC = yL_AC[...,self.resonant_dc_shift:-self.resonant_dc_shift]
                    idx = (np.arange(1, 2*self.nmax+1), np.arange(2*self.nmax))
                    self.yL.values[idx] = yL_AC[...,1:]
                    idx = (np.arange(0, 2*self.nmax), np.arange(1, 2*self.nmax+1))
                    self.yL.values[idx] = yL_AC[...,:-1]
        if self.simplified_initial_conditions:
            self.gammaL *= 0

        self.global_properties.energy = 1j*self.d

    def initialize(self, **solveopts):
        if self.unitary_transformation:
            self.initialize_Utransformed(**solveopts)
        else:
            self.initialize_untransformed(**solveopts)

    def run(self,
            ir_cutoff : 'IR cutoff of RG flow' = 0,
            forget_flow : 'do not store RG flow' = True,
            save_filename : 'save intermediate results: string containing %d for number of iterations' = '',
            save_iterations : 'number of iterations after which intermediate result should be saved' = 0,
            **solveopts : 'keyword arguments passed to solver',
            ):
        """
        Initialize and solve the RG equations.

        Arguments:
        ir_cutoff: Stop the RG flow at Λ = -iE = ir_cutoff (instead of
                Λ=0). If the RG flow is interrupted earlier, ir_cutoff
                will be adapted.
        **solveopts: keyword arguments passed to the solver. Most
                interesting are rtol and atol.

        1.  Get initial conditions for Γ, Z and G2 by numerically
            solving the equilibrium RG equations from E=0 to E=iD and
            for all required Re(E). Initialize G3, Iγ, δΓ, δΓγ, Yγ.
        2.  Solve RG equations from E=iD to E=0
            for Γ, Z, G2, G3, Iγ, δΓ, δΓγ, Yγ.
            Write parameters and solution for E=0 to self.<variables>
            Return the ODE solver.
        """
        self.initialize(**solveopts)

        self.save_filename = save_filename
        self.save_iterations = save_iterations

        if ir_cutoff:
            self.ir_cutoff = ir_cutoff
        self.solveopts = solveopts

        if ir_cutoff >= self.d:
            return

        ### Solve RG ODE
        output = self.solveOdeIm(self.d, ir_cutoff, only_final=forget_flow, **solveopts)

        # Write final values to Floquet matrices in self.
        try:
            # Shift energy
            self.global_properties.energy = self.energy.real + 1j*output.t[-1]
            # Unpack values
            self.unpackFlattenedValues(output.y[:,-1])
        except:
            settings.logger.exception("Failed to read solver results:")

        return output


    def updateRGequations(self):
        """
        Calculates the energy derivatives using the RG equations.
        The derivative of self.<name> is written to self.<name>E.

        A human readable reference implementation for this function
        is provided in updateRGequations_reference.
        """
        if settings.ENFORCE_SYMMETRIC:
            if hasattr(self, 'pi'):
                assert self.pi.symmetry == -1
            assert self.z.symmetry == 1
            assert self.yL.symmetry == -1
            assert self.gamma.symmetry == 1
            assert self.gammaL.symmetry == 1
            assert self.deltaGamma.symmetry == 1
            assert self.deltaGammaL.symmetry == 1
            assert self.g2[0,0].symmetry == 1
            assert self.g2[1,1].symmetry == 1
            assert self.g3[0,0].symmetry == -1
            assert self.g3[1,1].symmetry == -1
            assert self.current[0,0].symmetry == -1
            assert self.current[1,1].symmetry == -1
            if self.include_Ga:
                try:
                    assert self.ga_scalar.symmetry == -1
                except AttributeError:
                    assert self.ga[0,0].symmetry == -1
                    assert self.ga[1,1].symmetry == -1

        # Print some log message to indicate progress
        global LAST_LOG_TIME, REF_TIME
        if settings.LOG_TIME > 0 and time() - LAST_LOG_TIME >= settings.LOG_TIME:
            LAST_LOG_TIME = time()
            settings.logger.info(
                    "%9.2fs:  Λ = %.4e,  iterations = %d"%(
                        process_time(), self.energy.imag, self.total_iterations))

        if settings.USE_REFERENCE_IMPLEMENTATION:
            return self.updateRGequations_reference()

        symmetry = 0 if settings.IGNORE_SYMMETRIES else 1

        # Denote costs in terms of Floquet matrix products as x/y/z where
        # x is the number of multiplications for resonant_dc_shift != 0 without symmetry.
        # y is the number of multiplications for xL != xR but resonant_dc_shift == 0,
        # z is the number of multiplications for xL == xR,

        # Total costs: (+ 2 inversions, optionally + 2 multiplications for pi.derivative)
        # 178 / 116 / 67

        ## RG eq for Γ
        zinv = self.z.inverse() # costs: 1 inversion
        self.gammaE = -1j*( zinv - (SymRGfunction if self.compact else RGfunction)(self.global_properties, 'identity') )
        del zinv

        # Derivative of G2
        # First calculate Π (at E and shifted by multiples of vdc or mu):
        self.pi = (-1j*self.gamma).k2lambda(self.mu) # costs: 1 inversion

        # first terms of g2E:
        # 1/2  G13  Π  G32 ,
        # 1/2  G32  Π  G13
        g2_pi = self.g2 @ self.pi # costs: 4/3/2
        g2E1, g2E2 = product_combinations( g2_pi, self.g2 ) # costs: 12/7/4
        # g2E1.tr() = -i (d/dE)² Γ
        g2E1tr = g2E1.tr()
        g2E1tr.symmetry = -symmetry
        if self.compact and not isinstance(g2E1tr, SymRGfunction):
            g2E1tr = SymRGfunction.fromRGfunction(g2E1tr, diag=True, offdiag=False)
        g2E1 *= 0.5
        g2E2 *= 0.5
        self.zE = self.z @ g2E1tr @ self.z # costs: 2/2/2
        del g2E1tr

        ## RG eq for Ga
        if self.include_Ga:
            if hasattr(self, "ga_scalar"):
                self.ga_scalarE = g2E1[0,0] - g2E2[0,0]
            else:
                self.gaE = g2E1 - g2E2

        ## RG eq for G2
        self.g2E = g2E1 + g2E2
        if self.truncation_order >= 3:
            # third term for g2E
            # -1/4  G34 ( Π  G12  Z  +  Z  G12  Π ) G43
            # First the part inside the brackets:
            if self.solve_integral_exactly:
                # χ = Z (E + μ + NΩ + iΓ)
                chi = frequency_integral.calculate_chi(self.gamma, self.z)
                #             ⎛ ∞     1          1   ⎞
                # bracket = 2 ⎜ ∫dω ————— Z G² ————— ⎟ Z
                #             ⎝ 0   ω + χ      ω + χ ⎠
                pi_g2_z = 2 * frequency_integral.reservoir_matrix_integral(chi, self.z @ self.g2, chi, self.integral_method)
                pi_g2_z @= self.z
            else:
                pi_g2_z = self.pi @ self.g2 @ self.z + self.z @ g2_pi # costs: 12/9/6

            g2_bracket_g2, g2_bracket_g3 = einsum_34_12_43_double(self.g2, pi_g2_z, self.g2, self.g3) # costs: 48/30/15

            ## RG eq for G2
            self.g2E += (-0.25)*g2_bracket_g2
            del g2_bracket_g2

            if self.include_Ga:
                if hasattr(self, "ga_scalar"):
                    ga_pi2 = 2*self.ga_scalar @ self.pi
                    ga_pi_g2 = ga_pi2 @ self.g2[0,1] - 2*g2_pi[0,1] @ self.ga_scalar
                    ga_pi_g2_conj = ga_pi_g2.floquetConjugate()
                    ga_pi_g2_conj.voltage_shifts = -1
                    self.g2E[0,1] += ga_pi_g2
                    self.g2E[1,0] -= ga_pi_g2_conj
                    del ga_pi_g2, ga_pi_g2_conj
                else:
                    ga_pi = self.ga @ self.pi
                    g2_pi_ga_1, g2_pi_ga_2 = product_combinations( g2_pi, self.ga ) # costs: 12/7/4
                    ga_pi_g2_1, ga_pi_g2_2 = product_combinations( ga_pi, self.g2 ) # costs: 12/7/4
                    self.g2E += g2_pi_ga_1 + ga_pi_g2_1 - g2_pi_ga_2 - ga_pi_g2_2
        self.g2E.symmetry = -symmetry
        self.g2E[0,0].symmetry = -symmetry
        self.g2E[1,1].symmetry = -symmetry
        self.deltaGammaE = 1j * (g2E1[0,0] - g2E1[1,1] + g2E2[1,1] - g2E2[0,0]).reduced_to_voltage_branches(1)
        self.deltaGammaE.symmetry = -symmetry
        del g2E1, g2E2

        # first terms of G3E:
        # G2_13  Π  G3_32 ,
        # G2_32  Π  G3_13
        g3E1, g3E2 = product_combinations( g2_pi, self.g3 ) # costs: 12/7/4

        del g2_pi

        self.g3E = g3E1 + g3E2
        if self.truncation_order >= 3:
            # third term for g3E
            # 1/2  G2_34 ( Π  G2_12  Z  +  Z  G2_12  Π ) G3_43
            # -> already calculated
            self.g3E += 0.5 * g2_bracket_g3
            del g2_bracket_g3
            if self.include_Ga:
                if hasattr(self, "ga_scalar"):
                    ga_pi_g3 = ga_pi2 @ self.g3[0,1]
                    ga_pi_g3_conj = ga_pi_g3.floquetConjugate()
                    ga_pi_g3_conj.voltage_shifts = -1
                    self.g3E[0,1] += ga_pi_g3
                    self.g3E[1,0] += ga_pi_g3_conj
                    del ga_pi_g3, ga_pi_g3_conj, ga_pi2
                else:
                    ga_pi_g3_1, ga_pi_g3_2 = product_combinations( ga_pi, self.g3 ) # costs: 12/7/4
                    self.g3E += ga_pi_g3_1 - ga_pi_g3_2
        self.g3E.symmetry = symmetry
        self.g3E[0,0].symmetry = symmetry
        self.g3E[1,1].symmetry = symmetry
        del g3E1, g3E2


        # first terms of iE:
        # I13  Π  G32 ,
        # I32  Π  G13
        i_pi = self.current @ self.pi # costs: 4/3/2
        i_pi.symmetry = 0
        if settings.ENFORCE_SYMMETRIC:
            assert i_pi[0,0].symmetry == 1
            assert i_pi[1,1].symmetry == 1
        iE1, iE2 = product_combinations( i_pi, self.g2 ) # costs: 12/7/4

        self.currentE = iE1 + iE2
        del iE1, iE2
        if self.truncation_order >= 3:
            # third term for iE
            # 1/2  I34 ( Π  G2_12  Z  +  Z  G2_12  Π ) G43
            # The part in the brackets has been calculated before.
            self.currentE += 0.5 * einsum_34_12_43( self.current, pi_g2_z, self.g2 ) # costs: 32/20/10
            del pi_g2_z
            if self.include_Ga:
                if hasattr(self, "ga_scalar"):
                    i_pi_ga = -2*i_pi[0,1] @ self.ga_scalar
                    i_pi_ga_conj = i_pi_ga.floquetConjugate()
                    i_pi_ga_conj.voltage_shifts = -1
                    self.currentE[0,1] += i_pi_ga
                    self.currentE[1,0] += i_pi_ga_conj
                    del i_pi_ga, i_pi_ga_conj
                else:
                    il_pi_ga_1, il_pi_ga_2 = product_combinations( i_pi, self.ga ) # costs: 12/7/4
                    self.currentE += il_pi_ga_1 - il_pi_ga_2
        self.currentE.symmetry = symmetry
        self.currentE[0,0].symmetry = symmetry
        self.currentE[1,1].symmetry = symmetry


        # RG equation for δΓ_L
        # First part:  (δ1L - δ2L)  I_12  Π  G^3_21
        i_pi_g3_01 = i_pi[0,1] @ self.g3[1,0] # costs: 1
        i_pi_g3_10 = i_pi[1,0] @ self.g3[0,1] # costs: 1
        self.deltaGammaLE = 1.5j*(i_pi_g3_01 - i_pi_g3_10)
        # Second part:  I_12  Z  Π  δΓ  G^3_21
        if self.truncation_order >= 3:
            z_reduced = self.z.reduced_to_voltage_branches(1)
            dGamma_reduced = self.deltaGamma.reduced_to_voltage_branches(1)
            pi_reduced = self.pi.reduced_to_voltage_branches(1)
            if self.solve_integral_exactly:
                # TODO: check this!
                chi_reduced = chi.reduced_to_voltage_branches(1)
                z_dgamma_pi = 2*frequency_integral.floquet_matrix_integral(chi_reduced, z_reduced @ dGamma_reduced, chi_reduced, self.integral_method) @ z_reduced
            else:
                z_dgamma_pi = z_reduced @ dGamma_reduced @ pi_reduced \
                            + pi_reduced @ dGamma_reduced @ z_reduced # costs: 4/4/4
            del z_reduced, pi_reduced, dGamma_reduced
            if self.compact:
                assert isinstance(z_dgamma_pi, SymRGfunction)
            if settings.ENFORCE_SYMMETRIC:
                assert z_dgamma_pi.symmetry == -1
            self.deltaGammaLE += (-0.75)*einsum_34_12_43( self.current, z_dgamma_pi, self.g3 ) # costs: 32/20/10
            del z_dgamma_pi
        self.deltaGammaLE.symmetry = -symmetry


        # RG equation for ΓL
        self.gammaLE = self.yL
        self.yLE = 1.5j*(i_pi[0,0] @ self.g3[0,0] + i_pi[1,1] @ self.g3[1,1] + i_pi_g3_01 + i_pi_g3_10) # costs: 2/2/2
        self.yLE.symmetry = symmetry
        del i_pi, i_pi_g3_10, i_pi_g3_01

        if self.compact:
            # TODO: This is probably inefficient
            if not isinstance(self.deltaGammaE, SymRGfunction):
                self.deltaGammaE = SymRGfunction.fromRGfunction(self.deltaGammaE, diag=False, offdiag=True)
            if not isinstance(self.deltaGammaLE, SymRGfunction):
                self.deltaGammaLE = SymRGfunction.fromRGfunction(self.deltaGammaLE, diag=True, offdiag=False)
            if not isinstance(self.g2E[0,0], SymRGfunction):
                self.g2E[0,0] = SymRGfunction.fromRGfunction(self.g2E[0,0], diag=True, offdiag=False)
            if not isinstance(self.g2E[1,1], SymRGfunction):
                self.g2E[1,1] = SymRGfunction.fromRGfunction(self.g2E[1,1], diag=True, offdiag=False)
            if not isinstance(self.g3E[0,0], SymRGfunction):
                self.g3E[0,0] = SymRGfunction.fromRGfunction(self.g3E[0,0], diag=True, offdiag=False)
            if not isinstance(self.g3E[1,1], SymRGfunction):
                self.g3E[1,1] = SymRGfunction.fromRGfunction(self.g3E[1,1], diag=True, offdiag=False)
            if self.include_Ga:
                if hasattr(self, "ga_scalar"):
                    if not isinstance(self.ga_scalarE, SymRGfunction):
                        self.ga_scalarE = SymRGfunction.fromRGfunction(self.ga_scalarE, diag=True, offdiag=False)
                else:
                    if not isinstance(self.gaE[0,0], SymRGfunction):
                        self.gaE[0,0] = SymRGfunction.fromRGfunction(self.gaE[0,0], diag=True, offdiag=False)
                    if not isinstance(self.gaE[1,1], SymRGfunction):
                        self.gaE[1,1] = SymRGfunction.fromRGfunction(self.gaE[1,1], diag=True, offdiag=False)
            if not isinstance(self.currentE[0,0], SymRGfunction):
                self.currentE[0,0] = SymRGfunction.fromRGfunction(self.currentE[0,0], diag=False, offdiag=True)
            if not isinstance(self.currentE[1,1], SymRGfunction):
                self.currentE[1,1] = SymRGfunction.fromRGfunction(self.currentE[1,1], diag=False, offdiag=True)
            if not isinstance(self.yLE, SymRGfunction):
                self.yLE = SymRGfunction.fromRGfunction(self.yLE, diag=False, offdiag=True)

        # Count calls to RG equations.
        self.total_iterations += 1


    def updateRGequations_reference(self):
        """
        Reference implementation of updateRGequations without optimization.
        This reference implementation serves as a check for the more efficient
        function updateRGequations.

        This function takes approximately twice as long as updateRGequations.
        """
        # Notation (mainly allow using objects without "self")
        z = self.z
        gamma = self.gamma
        deltaGamma = self.deltaGamma
        deltaGammaL = self.deltaGammaL
        g2 = self.g2
        g3 = self.g3
        il = self.current
        yL = self.yL
        # Identity matrix
        identity = RGfunction(self.global_properties, 'identity')
        # Resolvent
        pi = self.pi = (-1j*gamma).k2lambda(self.mu)

        # Compute the sum
        #  Σ  A   B  C
        # 1,2  12     21
        einsum_34_x_43 = lambda a, b, c: \
                  a[0,0] @ b @ c[0,0] \
                + a[0,1] @ b @ c[1,0] \
                + a[1,0] @ b @ c[0,1] \
                + a[1,1] @ b @ c[1,1]

        ### RG equations in human readable form

        # Note that the shifts in the energy arguments of all RGfunction
        # objects is implicitly handled by the multiplication operators.
        # The muliplication operators for ReservoirMatrix objects are:
        # @ for normal matrix multiplication (matrix indices ij, jk → ik with sum over j)
        # % for transpose matrix multiplication (matrix indices jk, ij → ik with sum over j)
        #                 ⎛ ⊤   ⊤⎞⊤
        #   i.e.  A % B = ⎜A @ B ⎟  when there are no energy argument shifts.
        #                 ⎝      ⎠

        # dΓ      ⎛ 1     ⎞
        # —— = -i ⎜ — - 1 ⎟
        # dE      ⎝ Z     ⎠
        self.gammaE = -1j*(z.inverse() - identity)

        # dZ
        # —— = Z tr( G² Π G² ) Z
        # dE
        self.zE = z @ (g2 @ pi @ g2).tr() @ z

        if self.include_Ga:
            # dGa   1           1 ⎛  ⊤     ⊤⎞⊤
            # ——— = — G² Π G² - — ⎜G²  Π G² ⎟
            # dE    2           2 ⎝         ⎠
            self.gaE = .5 * g2 @ pi @ g2 - .5 * ((g2 @ pi) % g2)

        if self.truncation_order == 3:
            if settings.EXTRAPOLATE_VOLTAGE:
                raise NotImplementedError
            if self.solve_integral_exactly:
                # χ = Z (E + μ + NΩ + iΓ)
                chi = frequency_integral.calculate_chi(self.gamma, self.z)
                #             ⎛ ∞     1          1   ⎞
                # bracket = 2 ⎜ ∫dω ————— Z G² ————— ⎟ Z
                #             ⎝ 0   ω + χ      ω + χ ⎠
                bracket = 2 * frequency_integral.reservoir_matrix_integral(chi, self.z @ self.g2, chi, self.integral_method)
                bracket @= z
            else:
                # bracket = Π G² Z + Z G² Π
                bracket = pi @ g2 @ z + z @ g2 @ pi

            # dG²   1           1 ⎛  ⊤     ⊤⎞⊤   1     ⎛                 ⎞
            # ——— = — G² Π G² + — ⎜G²  Π G² ⎟  - — G²  ⎜ Π G² Z + Z G² Π ⎟ G²
            # dE    2           2 ⎝         ⎠    4  34 ⎝                 ⎠  43
            self.g2E = .5 * g2 @ pi @ g2 + .5 * ((g2 @ pi) % g2) - .25 * einsum_34_x_43(g2, bracket, g2)

            # dG³             ⎛  ⊤     ⊤⎞⊤   1     ⎛                 ⎞
            # ——— = G² Π G³ + ⎜G²  Π G³ ⎟  + — G²  ⎜ Π G² Z + Z G² Π ⎟ G³
            # dE              ⎝         ⎠    2  34 ⎝                 ⎠  43
            self.g3E = g2 @ pi @ g3 + ((g2 @ pi) % g3) + .5 * einsum_34_x_43(g2, bracket, g3)

            #   γ
            # dI     γ        ⎛ γ⊤     ⊤⎞⊤   1  γ  ⎛                 ⎞
            # ——— = I  Π G² + ⎜I   Π G² ⎟  + — I   ⎜ Π G² Z + Z G² Π ⎟ G²
            # dE              ⎝         ⎠    2  34 ⎝                 ⎠  43
            self.currentE = il @ pi @ g2 + ((il @ pi) % g2) + .5 * einsum_34_x_43(il, bracket, g2)

            if self.include_Ga:
                ga = self.ga

                # dG²              ⎛  ⊤     ⊤⎞⊤             ⎛  ⊤     ⊤⎞⊤
                # ——— += G² Π Ga - ⎜G²  Π Ga ⎟  + Ga Π G² - ⎜Ga  Π G² ⎟
                # dE               ⎝         ⎠              ⎝         ⎠
                self.g2E += g2 @ pi @ ga + ga @ pi @ g2 - ((g2 @ pi) % ga) - ((ga @ pi) % g2)

                # dG³              ⎛  ⊤     ⊤⎞⊤
                # ——— += Ga Π G³ - ⎜Ga  Π G³ ⎟
                # dE               ⎝         ⎠
                self.g3E += ga @ pi @ g3 - ((ga @ pi) % g3)

                #   γ
                # dI      γ        ⎛ γ⊤     ⊤⎞⊤
                # ——— += I  Π Ga - ⎜I   Π Ga ⎟
                # dE               ⎝         ⎠
                self.currentE += il @ pi @ ga - ((il @ pi) % ga)

        elif self.tuncation_order == 2:
            # dG²   1           1 ⎛  ⊤     ⊤⎞⊤
            # ——— = — G² Π G² + — ⎜G²  Π G² ⎟
            # dE    2           2 ⎝         ⎠
            self.g2E = .5 * g2 @ pi @ g2 + .5 * ((g2 @ pi) % g2)

            # dG³             ⎛  ⊤     ⊤⎞⊤
            # ——— = G² Π G³ + ⎜G²  Π G³ ⎟
            # dE              ⎝         ⎠
            self.g3E = g2 @ pi @ g3 + ((g2 @ pi) % g3)

            #   γ
            # dI     γ        ⎛ γ⊤     ⊤⎞⊤
            # ——— = I  Π G² + ⎜I   Π G² ⎟
            # dE              ⎝         ⎠
            self.currentE = il @ pi @ g2 + ((il @ pi) % g2)

        else:
            raise ValueError("Invalid truncation order: %s"%self.truncation_order)

        # dδΓ     ⎛           ⎞
        # ——— = i ⎜ δ  - δ    ⎟ G²  Π G²
        # dE      ⎝  1L    2L ⎠  12    21
        deltaGammaE = 1j * (g2[0,1] @ pi @ g2[1,0] - g2[1,0] @ pi @ g2[0,1])

        # Reduction of voltage branches as required for the solver.
        # In this step some information is thrown away that cannot affect the
        # result of the physical observables.
        self.deltaGammaE = deltaGammaE.reduced_to_voltage_branches(1)
        pi_reduced = pi.reduced_to_voltage_branches(1)
        z_reduced = z.reduced_to_voltage_branches(1)

        #    γ
        # dδΓ    3   ⎛           ⎞  γ
        # ———— = — i ⎜ δ  - δ    ⎟ I   Π G³
        #  dE    2   ⎝  1L    2L ⎠  12    21
        self.deltaGammaLE = 1.5j * (il[0,1] @ pi @ g3[1,0] - il[1,0] @ pi @ g3[0,1])

        if self.truncation_order == 3:
            if self.solve_integral_exactly:
                # TODO: check this!
                chi_reduced = chi.reduced_to_voltage_branches(1)
                integral_result = frequency_integral.floquet_matrix_integral(chi_reduced, z_reduced @ deltaGamma, chi_reduced, self.integral_method) @ z_reduced
                self.deltaGammaLE += (-1.5) * (il @ integral_result @ g3).tr()
            else:
                #    γ
                # dδΓ     3    ⎛ γ⎛                 ⎞   ⎞
                # ———— -= — tr ⎜I ⎜ Π δΓ Z + Z δΓ Π ⎟ G³⎟
                #  dE     4    ⎝  ⎝                 ⎠   ⎠
                self.deltaGammaLE += (-0.75) * (il @ (pi_reduced @ deltaGamma @ z_reduced + z_reduced @ deltaGamma @ pi_reduced) @ g3).tr()

        #   γ
        # dΓ     γ
        # ——— = Y
        # dE
        self.gammaLE = yL

        #   γ
        # dY    3      ⎛  γ      ⎞
        # ——— = — i tr ⎜ I  Π G³ ⎟
        # dE    2      ⎝         ⎠
        self.yLE = 1.5j * (il @ pi @ g3).tr()

        # Count calls to RG equations.
        self.total_iterations += 1


    def updateRGequationsMinimal(self):
        """
        Calculates the energy derivatives using the RG equations.
        Only include Gamma, Z, G2, Ga.
        The derivative of self.<name> is written to self.<name>E.
        """
        # TODO: update handling of Ga
        if settings.ENFORCE_SYMMETRIC:
            if hasattr(self, 'pi'):
                assert self.pi.symmetry == -1
            assert self.z.symmetry == 1
            assert self.gamma.symmetry == 1
            assert self.g2[0,0].symmetry == 1
            assert self.g2[1,1].symmetry == 1
            if self.include_Ga:
                assert self.ga[0,0].symmetry == -1
                assert self.ga[1,1].symmetry == -1

        # Print some log message to indicate progress
        global LAST_LOG_TIME, REF_TIME
        if settings.LOG_TIME > 0 and time() - LAST_LOG_TIME >= settings.LOG_TIME:
            LAST_LOG_TIME = time()
            settings.logger.info(
                    "%9.2fs:  Λ = %.4e,  iterations = %d"%(
                        process_time(), self.energy.imag, self.total_iterations))

        if settings.USE_REFERENCE_IMPLEMENTATION:
            settings.logger.warn("There is no reference implementation for minimal RG equations. Using default implementation.")

        symmetry = 0 if settings.IGNORE_SYMMETRIES else 1

        ## RG eq for Γ
        zinv = self.z.inverse() # costs: 1 inversion
        self.gammaE = -1j*( zinv - (SymRGfunction if self.compact else RGfunction)(self.global_properties, 'identity') )
        del zinv

        # Derivative of G2
        # First calculate Π (at E and shifted by multiples of vdc or mu):
        self.pi = (-1j*self.gamma).k2lambda(self.mu) # costs: 1 inversion

        # first terms of g2E:
        # 1/2  G13  Π  G32 ,
        # 1/2  G32  Π  G13
        g2_pi = self.g2 @ self.pi # costs: 4/3/2
        g2E1, g2E2 = product_combinations( g2_pi, self.g2 ) # costs: 12/7/4
        # g2E1.tr() = -i (d/dE)² Γ
        g2E1tr = g2E1.tr()
        g2E1tr.symmetry = -symmetry
        if self.compact and not isinstance(g2E1tr, SymRGfunction):
            g2E1tr = SymRGfunction.fromRGfunction(g2E1tr, diag=True, offdiag=False)
        g2E1 *= 0.5
        g2E2 *= 0.5
        self.zE = self.z @ g2E1tr @ self.z # costs: 2/2/2
        del g2E1tr

        ## RG eq for Ga
        if self.include_Ga:
            if hasattr(self, "ga_scalar"):
                self.ga_scalarE = g2E1[0,0] - g2E2[0,0]
            else:
                self.gaE = g2E1 - g2E2

        ## RG eq for G2
        self.g2E = g2E1 + g2E2
        if self.truncation_order >= 3:
            # third term for g2E
            # -1/4  G34 ( Π  G12  Z  +  Z  G12  Π ) G43
            # First the part inside the brackets:
            if self.solve_integral_exactly:
                # χ = Z (E + μ + NΩ + iΓ)
                chi = frequency_integral.calculate_chi(self.gamma, self.z)
                #             ⎛ ∞     1          1   ⎞
                # bracket = 2 ⎜ ∫dω ————— Z G² ————— ⎟ Z
                #             ⎝ 0   ω + χ      ω + χ ⎠
                pi_g2_z = 2 * frequency_integral.reservoir_matrix_integral(chi, self.z @ self.g2, chi, self.integral_method)
                pi_g2_z @= self.z
            else:
                pi_g2_z = self.pi @ self.g2 @ self.z + self.z @ g2_pi # costs: 12/9/6

            ## RG eq for G2
            self.g2E += (-0.25)*einsum_34_12_43(self.g2, pi_g2_z, self.g2)
            del pi_g2_z

            if self.include_Ga:
                if hasattr(self, "ga_scalar"):
                    ga_pi2 = 2*self.ga_scalar @ self.pi
                    ga_pi_g2 = ga_pi2 @ self.g2[0,1] - 2*g2_pi[0,1] @ self.ga_scalar
                    ga_pi_g2_conj = ga_pi_g2.floquetConjugate()
                    ga_pi_g2_conj.voltage_shifts = -1
                    self.g2E[0,1] += ga_pi_g2
                    self.g2E[1,0] -= ga_pi_g2_conj
                    del ga_pi_g2, ga_pi_g2_conj
                else:
                    ga_pi = self.ga @ self.pi
                    g2_pi_ga_1, g2_pi_ga_2 = product_combinations( g2_pi, self.ga ) # costs: 12/7/4
                    ga_pi_g2_1, ga_pi_g2_2 = product_combinations( ga_pi, self.g2 ) # costs: 12/7/4
                    self.g2E += g2_pi_ga_1 + ga_pi_g2_1 - g2_pi_ga_2 - ga_pi_g2_2
        self.g2E.symmetry = -symmetry
        self.g2E[0,0].symmetry = -symmetry
        self.g2E[1,1].symmetry = -symmetry
        del g2E1, g2E2, g2_pi

        if self.compact:
            if not isinstance(self.g2E[0,0], SymRGfunction):
                self.g2E[0,0] = SymRGfunction.fromRGfunction(self.g2E[0,0], diag=True, offdiag=False)
            if not isinstance(self.g2E[1,1], SymRGfunction):
                self.g2E[1,1] = SymRGfunction.fromRGfunction(self.g2E[1,1], diag=True, offdiag=False)
            if self.include_Ga:
                if hasattr(self, "ga_scalar"):
                    if not isinstance(self.ga_scalarE, SymRGfunction):
                        self.ga_scalarE = SymRGfunction.fromRGfunction(self.ga_scalarE, diag=True, offdiag=False)
                else:
                    if not isinstance(self.gaE[0,0], SymRGfunction):
                        self.gaE[0,0] = SymRGfunction.fromRGfunction(self.gaE[0,0], diag=True, offdiag=False)
                    if not isinstance(self.gaE[1,1], SymRGfunction):
                        self.gaE[1,1] = SymRGfunction.fromRGfunction(self.gaE[1,1], diag=True, offdiag=False)

        # Count calls to RG equations.
        self.total_iterations += 1


    def check_symmetry(self):
        """
        Check if all symmetries are fulfilled
        """
        self.gamma.check_symmetry()
        self.gammaL.check_symmetry()
        self.deltaGamma.check_symmetry()
        self.deltaGammaL.check_symmetry()
        self.z.check_symmetry()
        self.yL.check_symmetry()
        self.g2.check_symmetry()
        self.g3.check_symmetry()
        if self.include_Ga:
            self.ga.check_symmetry()
        self.current.check_symmetry()


    def unpackFlattenedValues(self, flattened_values):
        """
        Translate between 1d array used by the solver and Floquet matrices
        used in RG equations. Given a 1d array, write the values of this array
        to the Floquet matrices self.<values>.

        Order of flattened_values:
        Γ, Z, δΓ, *G2, *G3, *IL, δΓL, ΓL, YL
        """
        if self.compact == 0:
            s = self.yL.values.size
            m = self.deltaGamma.values.size
            l = self.z.values.size
            if self.include_Ga:
                if hasattr(self, "ga_scalar"):
                    assert flattened_values.size == 11*l+m+7*s
                else:
                    assert flattened_values.size == 14*l+m+7*s
            else:
                assert flattened_values.size == 10*l+m+7*s
            self.gamma.values = flattened_values[:l].reshape(self.gamma.values.shape)
            self.z.values = flattened_values[l:2*l].reshape(self.z.values.shape)
            self.deltaGamma.values = flattened_values[2*l:2*l+m].reshape(self.deltaGamma.values.shape)
            for (g2i, flat) in zip(self.g2.data.flat, np.split(flattened_values[2*l+m:6*l+m], 4)):
                g2i.values = flat.reshape(g2i.values.shape)
            for (g3i, flat) in zip(self.g3.data.flat, np.split(flattened_values[6*l+m:10*l+m], 4)):
                g3i.values = flat.reshape(g3i.values.shape)
            for (ii, flat) in zip(self.current.data.flat, np.split(flattened_values[10*l+m:10*l+m+4*s], 4)):
                ii.values = flat.reshape(ii.values.shape)
            self.deltaGammaL.values = flattened_values[10*l+m+4*s:10*l+m+5*s].reshape(self.deltaGammaL.values.shape)
            self.gammaL.values = flattened_values[10*l+m+5*s:10*l+m+6*s].reshape(self.gammaL.values.shape)
            self.yL.values = flattened_values[10*l+m+6*s:10*l+m+7*s].reshape(self.yL.values.shape)
            if self.include_Ga:
                if flattened_values.size == 11*l+m+7*s:
                    self.ga_scalar.values = flattened_values[10*l+m+7*s:11*l+m+7*s].reshape(self.ga_scalar.values.shape)
                else:
                    for (gai, flat) in zip(self.ga.data.flat, np.split(flattened_values[10*l+m+7*s:14*l+m+7*s], 4)):
                        gai.values = flat.reshape(gai.values.shape)
        elif self.compact == 1:
            nmax = self.nmax
            f = (2*nmax+1)**2
            o = (nmax+1)**2
            i = nmax**2
            m = nmax*(nmax+1)
            shape_f = (2*nmax+1, 2*nmax+1)
            shape00 = (nmax+1, nmax+1)
            shape11 = (nmax, nmax)
            shape01 = (nmax+1, nmax)
            shape10 = (nmax, nmax+1)
            if self.include_Ga and hasattr(self, "ga_scalar"):
                assert flattened_values.size == 8*(o+i) + 10*m + 6*f
            elif self.include_Ga:
                assert flattened_values.size == 9*(o+i) + 10*m + 8*f
            else:
                assert flattened_values.size == 7*(o+i) + 10*m + 6*f
            self.gamma.submatrix00 = flattened_values[:o].reshape(shape00)
            self.z.submatrix00 = flattened_values[o:2*o].reshape(shape00)
            self.deltaGammaL.submatrix00 = flattened_values[2*o:3*o].reshape(shape00)
            self.g2[0,0].submatrix00 = flattened_values[3*o:4*o].reshape(shape00)
            self.g2[1,1].submatrix00 = flattened_values[4*o:5*o].reshape(shape00)
            self.g3[0,0].submatrix00 = flattened_values[5*o:6*o].reshape(shape00)
            self.g3[1,1].submatrix00 = flattened_values[6*o:7*o].reshape(shape00)
            self.gamma.submatrix11 = flattened_values[7*o:7*o+i].reshape(shape11)
            self.z.submatrix11 = flattened_values[7*o+i:7*o+2*i].reshape(shape11)
            self.deltaGammaL.submatrix11 = flattened_values[7*o+2*i:7*o+3*i].reshape(shape11)
            self.g2[0,0].submatrix11 = flattened_values[7*o+3*i:7*o+4*i].reshape(shape11)
            self.g2[1,1].submatrix11 = flattened_values[7*o+4*i:7*o+5*i].reshape(shape11)
            self.g3[0,0].submatrix11 = flattened_values[7*o+5*i:7*o+6*i].reshape(shape11)
            self.g3[1,1].submatrix11 = flattened_values[7*o+6*i:7*(o+i)].reshape(shape11)
            self.gammaL.submatrix01 = flattened_values[7*(o+i):7*(o+i)+m].reshape(shape01)
            self.gammaL.submatrix10 = flattened_values[7*(o+i)+m:7*(o+i)+2*m].reshape(shape10)
            self.deltaGamma.submatrix01 = flattened_values[7*(o+i)+2*m:7*(o+i)+3*m].reshape(shape01)
            self.deltaGamma.submatrix10 = flattened_values[7*(o+i)+3*m:7*(o+i)+4*m].reshape(shape10)
            self.yL.submatrix01 = flattened_values[7*(o+i)+4*m:7*(o+i)+5*m].reshape(shape01)
            self.yL.submatrix10 = flattened_values[7*(o+i)+5*m:7*(o+i)+6*m].reshape(shape10)
            self.current[0,0].submatrix01 = flattened_values[7*(o+i)+6*m:7*(o+i+m)].reshape(shape01)
            self.current[0,0].submatrix10 = flattened_values[7*(o+i+m):7*(o+i)+8*m].reshape(shape10)
            self.current[1,1].submatrix01 = flattened_values[7*(o+i)+8*m:7*(o+i)+9*m].reshape(shape01)
            self.current[1,1].submatrix10 = flattened_values[7*(o+i)+9*m:7*(o+i)+10*m].reshape(shape10)
            self.g2[0,1].values = flattened_values[7*(o+i)+10*m:7*(o+i)+10*m+f].reshape(shape_f)
            self.g2[1,0].values = flattened_values[7*(o+i)+10*m+f:7*(o+i)+10*m+2*f].reshape(shape_f)
            self.g3[0,1].values = flattened_values[7*(o+i)+10*m+2*f:7*(o+i)+10*m+3*f].reshape(shape_f)
            self.g3[1,0].values = flattened_values[7*(o+i)+10*m+3*f:7*(o+i)+10*m+4*f].reshape(shape_f)
            self.current[0,1].values = flattened_values[7*(o+i)+10*m+4*f:7*(o+i)+10*m+5*f].reshape(shape_f)
            self.current[1,0].values = flattened_values[7*(o+i)+10*m+5*f:7*(o+i)+10*m+6*f].reshape(shape_f)
            if self.include_Ga:
                if flattened_values.size == 8*(o+i) + 10*m + 6*f:
                    self.ga_scalar.submatrix00 = flattened_values[7*o+7*i+10*m+6*f:8*o+7*i+10*m+6*f].reshape(shape00)
                    self.ga_scalar.submatrix11 = flattened_values[8*o+7*i+10*m+6*f:8*o+8*i+10*m+6*f].reshape(shape11)
                else:
                    self.ga[0,0].submatrix00 = flattened_values[7*o+7*i+10*m+6*f:8*o+7*i+10*m+6*f].reshape(shape00)
                    self.ga[1,1].submatrix00 = flattened_values[8*o+7*i+10*m+6*f:9*o+7*i+10*m+6*f].reshape(shape00)
                    self.ga[0,0].submatrix11 = flattened_values[9*o+7*i+10*m+6*f:9*o+8*i+10*m+6*f].reshape(shape11)
                    self.ga[1,1].submatrix11 = flattened_values[9*o+8*i+10*m+6*f:9*o+9*i+10*m+6*f].reshape(shape11)
                    self.ga[0,1].values = flattened_values[9*o+9*i+10*m+6*f:9*o+9*i+10*m+7*f].reshape(shape_f)
                    self.ga[1,0].values = flattened_values[9*o+9*i+10*m+7*f:9*o+9*i+10*m+8*f].reshape(shape_f)
        elif self.compact == 2:
            nmax = self.nmax
            f = (2*nmax+1)**2
            o = ((nmax+1)**2 + 1) // 2
            i = (nmax**2 + 1) // 2
            m = nmax*(nmax+1) // 2
            if self.include_Ga:
                assert flattened_values.size == 6*(o+i) + 8*m + 3*f
            else:
                assert flattened_values.size == 5*(o+i) + 8*m + 3*f
            unpack01 = lambda flat, sym: np.concatenate((flat, sym*flat[::-1].conjugate()), axis=None).reshape((nmax+1,nmax))
            unpack10 = lambda flat, sym: np.concatenate((flat, sym*flat[::-1].conjugate()), axis=None).reshape((nmax,nmax+1))
            if nmax % 2:
                unpack00 = lambda flat, sym: np.concatenate((flat, sym*flat[::-1].conjugate()), axis=None).reshape((nmax+1,nmax+1))
                unpack11 = lambda flat, sym: np.concatenate((flat, sym*flat[-2::-1].conjugate()), axis=None).reshape((nmax,nmax))
            else:
                unpack00 = lambda flat, sym: np.concatenate((flat, sym*flat[-2::-1].conjugate()), axis=None).reshape((nmax+1,nmax+1))
                unpack11 = lambda flat, sym: np.concatenate((flat, sym*flat[::-1].conjugate()), axis=None).reshape((nmax,nmax))
            self.gamma.submatrix00 = unpack00(flattened_values[:o], 1)
            self.gamma.submatrix11 = unpack11(flattened_values[5*o:5*o+i], 1)
            self.z.submatrix00 = unpack00(flattened_values[o:2*o], 1)
            self.z.submatrix11 = unpack11(flattened_values[5*o+i:5*o+2*i], 1)
            self.deltaGammaL.submatrix00 = unpack00(flattened_values[2*o:3*o], 1)
            self.deltaGammaL.submatrix11 = unpack11(flattened_values[5*o+2*i:5*o+3*i], 1)
            self.g2[0,0].submatrix00 = unpack00(flattened_values[3*o:4*o], 1)
            self.g2[0,0].submatrix11 = unpack11(flattened_values[5*o+3*i:5*o+4*i], 1)
            self.g2[1,1] = self.g2[0,0].copy()
            self.g3[0,0].submatrix00 = unpack00(flattened_values[4*o:5*o], -1)
            self.g3[0,0].submatrix11 = unpack11(flattened_values[5*o+4*i:5*(o+i)], -1)
            self.g3[1,1] = self.g3[0,0].copy()
            self.gammaL.submatrix01 = unpack01(flattened_values[5*(o+i):5*(o+i)+m], 1)
            self.gammaL.submatrix10 = unpack10(flattened_values[5*(o+i)+m:5*(o+i)+2*m], 1)
            self.deltaGamma.submatrix01 = unpack01(flattened_values[5*(o+i)+2*m:5*(o+i)+3*m], 1)
            self.deltaGamma.submatrix10 = unpack10(flattened_values[5*(o+i)+3*m:5*(o+i)+4*m], 1)
            self.yL.submatrix01 = unpack01(flattened_values[5*(o+i)+4*m:5*(o+i+m)], -1)
            self.yL.submatrix10 = unpack10(flattened_values[5*(o+i+m):5*(o+i)+6*m], -1)
            self.current[0,0].submatrix01 = unpack01(flattened_values[5*(o+i)+6*m:5*(o+i)+7*m], -1)
            self.current[0,0].submatrix10 = unpack10(flattened_values[5*(o+i)+7*m:5*(o+i)+8*m], -1)
            self.current[1,1] = self.current[0,0].copy()
            self.g2[0,1].values = flattened_values[5*(o+i)+8*m:5*(o+i)+8*m+f].reshape((2*nmax+1, 2*nmax+1))
            self.g2[1,0] = self.g2[0,1].floquetConjugate()
            self.g3[0,1].values = flattened_values[5*(o+i)+8*m+f:5*(o+i)+8*m+2*f].reshape((2*nmax+1, 2*nmax+1))
            self.g3[1,0] = -self.g3[0,1].floquetConjugate()
            self.current[0,1].values = flattened_values[5*(o+i)+8*m+2*f:5*(o+i)+8*m+3*f].reshape((2*nmax+1, 2*nmax+1))
            self.current[1,0] = -self.current[0,1].floquetConjugate()
            if self.include_Ga:
                self.ga_scalar.submatrix00 = unpack00(flattened_values[5*(o+i)+8*m+3*f:6*o+5*i+8*m+3*f], -1)
                self.ga_scalar.submatrix11 = unpack11(flattened_values[6*o+5*i+8*m+3*f:6*(o+i)+8*m+3*f], -1)


        if settings.CHECK_SYMMETRIES:
            self.check_symmetry()


    def unpackFlattenedValuesMinimal(self, flattened_values):
        """
        Translate between 1d array used by the solver and Floquet matrices
        used in RG equations. Given a 1d array, write the values of this array
        to the Floquet matrices self.<values>.

        Order of flattened_values:
        Γ, Z, *G2, *Ga
        """
        if self.compact == 0:
            l = self.z.values.size
            if self.include_Ga:
                if hasattr(self, "ga_scalar"):
                    assert flattened_values.size == 7*l
                else:
                    assert flattened_values.size == 10*l
            else:
                assert flattened_values.size == 6*l
            self.gamma.values = flattened_values[:l].reshape(self.gamma.values.shape)
            self.z.values = flattened_values[l:2*l].reshape(self.z.values.shape)
            for (g2i, flat) in zip(self.g2.data.flat, np.split(flattened_values[2*l:6*l], 4)):
                g2i.values = flat.reshape(g2i.values.shape)
            if self.include_Ga:
                if flattened_values.size == 7*l:
                    self.ga_scalar.values = flattened_values[6*l:7*l].reshape(self.ga_scalar.values.shape)
                else:
                    for (gai, flat) in zip(self.ga.data.flat, np.split(flattened_values[6*l:10*l], 4)):
                        gai.values = flat.reshape(gai.values.shape)
        elif self.compact == 1:
            nmax = self.nmax
            f = (2*nmax+1)**2
            o = (nmax+1)**2
            i = nmax**2
            shape_f = (2*nmax+1, 2*nmax+1)
            shape00 = (nmax+1, nmax+1)
            shape11 = (nmax, nmax)
            shape01 = (nmax+1, nmax)
            shape10 = (nmax, nmax+1)
            if self.include_Ga:
                assert flattened_values.size == 6*(o+i) + 4*f
            else:
                assert flattened_values.size == 4*(o+i) + 2*f
            self.gamma.submatrix00 = flattened_values[:o].reshape(shape00)
            self.z.submatrix00 = flattened_values[o:2*o].reshape(shape00)
            self.g2[0,0].submatrix00 = flattened_values[2*o:3*o].reshape(shape00)
            self.g2[1,1].submatrix00 = flattened_values[3*o:4*o].reshape(shape00)
            self.gamma.submatrix11 = flattened_values[4*o:4*o+i].reshape(shape11)
            self.z.submatrix11 = flattened_values[4*o+i:4*o+2*i].reshape(shape11)
            self.g2[0,0].submatrix11 = flattened_values[4*o+2*i:4*o+3*i].reshape(shape11)
            self.g2[1,1].submatrix11 = flattened_values[4*o+3*i:4*o+4*i].reshape(shape11)
            self.g2[0,1].values = flattened_values[4*(o+i):4*(o+i)+f].reshape(shape_f)
            self.g2[1,0].values = flattened_values[4*(o+i)+f:4*(o+i)+2*f].reshape(shape_f)
            if self.include_Ga:
                self.ga[0,0].submatrix00 = flattened_values[4*o+4*i+2*f:5*o+4*i+2*f].reshape(shape00)
                self.ga[1,1].submatrix00 = flattened_values[5*o+4*i+2*f:6*o+4*i+2*f].reshape(shape00)
                self.ga[0,0].submatrix11 = flattened_values[6*o+4*i+2*f:6*o+5*i+2*f].reshape(shape11)
                self.ga[1,1].submatrix11 = flattened_values[6*o+5*i+2*f:6*o+6*i+2*f].reshape(shape11)
                self.ga[0,1].values = flattened_values[6*(o+i)+2*f:6*(o+i)+3*f].reshape(shape_f)
                self.ga[1,0].values = flattened_values[6*(o+i)+3*f:6*(o+i)+4*f].reshape(shape_f)
        elif self.compact == 2:
            nmax = self.nmax
            f = (2*nmax+1)**2
            o = ((nmax+1)**2 + 1) // 2
            i = (nmax**2 + 1) // 2
            if self.include_Ga:
                raise NotImplementedError
            assert flattened_values.size == 3*(o+i) + f
            unpack01 = lambda flat, sym: np.concatenate((flat, sym*flat[::-1].conjugate()), axis=None).reshape((nmax+1,nmax))
            unpack10 = lambda flat, sym: np.concatenate((flat, sym*flat[::-1].conjugate()), axis=None).reshape((nmax,nmax+1))
            if nmax % 2:
                unpack00 = lambda flat, sym: np.concatenate((flat, sym*flat[::-1].conjugate()), axis=None).reshape((nmax+1,nmax+1))
                unpack11 = lambda flat, sym: np.concatenate((flat, sym*flat[-2::-1].conjugate()), axis=None).reshape((nmax,nmax))
            else:
                unpack00 = lambda flat, sym: np.concatenate((flat, sym*flat[-2::-1].conjugate()), axis=None).reshape((nmax+1,nmax+1))
                unpack11 = lambda flat, sym: np.concatenate((flat, sym*flat[::-1].conjugate()), axis=None).reshape((nmax,nmax))
            self.gamma.submatrix00 = unpack00(flattened_values[:o], 1)
            self.z.submatrix00 = unpack00(flattened_values[o:2*o], 1)
            self.g2[0,0].submatrix00 = unpack00(flattened_values[2*o:3*o], 1)
            self.gamma.submatrix11 = unpack11(flattened_values[3*o:3*o+i], 1)
            self.z.submatrix11 = unpack11(flattened_values[3*o+i:3*o+2*i], 1)
            self.g2[0,0].submatrix11 = unpack11(flattened_values[3*o+2*i:3*(o+i)], 1)
            self.g2[1,1] = self.g2[0,0].copy()
            self.g2[0,1].values = flattened_values[3*(o+i):3*(o+i)+f].reshape((2*nmax+1, 2*nmax+1))
            self.g2[1,0] = self.g2[0,1].floquetConjugate()

        if settings.CHECK_SYMMETRIES:
            self.check_symmetry()


    def packFlattenedDerivatives(self):
        """
        Pack Floquet matrices representing derivatives in one flattened
        (1d) array for the solver.

        Order of flattened_values:
        Γ, Z, δΓ, *G2, *G3, *IL, δΓL, ΓL, YL
        """
        if self.compact == 0:
            all_data = [
                        self.gammaE.values,
                        self.zE.values,
                        self.deltaGammaE.values,
                        *(g.values for g in self.g2E.data.flat),
                        *(g.values for g in self.g3E.data.flat),
                        *(i.values for i in self.currentE.data.flat),
                        self.deltaGammaLE.values,
                        self.gammaLE.values,
                        self.yLE.values,
                    ]
            if self.include_Ga:
                try:
                    all_data.append(self.ga_scalarE.values)
                except AttributeError:
                    all_data += [*(g.values for g in self.gaE.data.flat)]
            return np.concatenate(all_data, axis=None)
        elif self.compact == 1:
            if settings.CHECK_SYMMETRIES:
                assert self.gammaE.submatrix01 is None
                assert self.gammaE.submatrix10 is None
                assert self.zE.submatrix01 is None
                assert self.zE.submatrix10 is None
                assert self.deltaGammaLE.submatrix01 is None
                assert self.deltaGammaLE.submatrix10 is None
                assert self.gammaLE.submatrix00 is None
                assert self.gammaLE.submatrix11 is None
                assert self.deltaGammaE.submatrix00 is None
                assert self.deltaGammaE.submatrix11 is None
                assert self.yLE.submatrix00 is None
                assert self.yLE.submatrix11 is None
                assert self.g2E[0,0].submatrix01 is None
                assert self.g2E[1,1].submatrix01 is None
                assert self.g3E[0,0].submatrix01 is None
                assert self.g3E[1,1].submatrix01 is None
                assert self.g2E[0,0].submatrix10 is None
                assert self.g2E[1,1].submatrix10 is None
                assert self.g3E[0,0].submatrix10 is None
                assert self.g3E[1,1].submatrix10 is None
                assert self.currentE[0,0].submatrix00 is None
                assert self.currentE[0,0].submatrix11 is None
                assert self.currentE[1,1].submatrix00 is None
                assert self.currentE[1,1].submatrix11 is None
            if self.yLE.submatrix01 is None:
                dummy = np.zeros(self.nmax*(self.nmax+1), dtype=np.complex128)
                self.yLE.submatrix01 = dummy
                self.yLE.submatrix10 = dummy
                if self.deltaGammaE.submatrix01 is None:
                    self.deltaGammaE.submatrix01 = dummy
                    self.deltaGammaE.submatrix10 = dummy
                if self.currentE[0,0].submatrix01 is None:
                    self.currentE[0,0].submatrix01 = dummy
                    self.currentE[0,0].submatrix10 = dummy
                if self.currentE[1,1].submatrix01 is None:
                    self.currentE[1,1].submatrix01 = dummy
                    self.currentE[1,1].submatrix10 = dummy
            all_data = [
                        self.gammaE.submatrix00,
                        self.zE.submatrix00,
                        self.deltaGammaLE.submatrix00,
                        self.g2E[0,0].submatrix00,
                        self.g2E[1,1].submatrix00,
                        self.g3E[0,0].submatrix00,
                        self.g3E[1,1].submatrix00,
                        self.gammaE.submatrix11,
                        self.zE.submatrix11,
                        self.deltaGammaLE.submatrix11,
                        self.g2E[0,0].submatrix11,
                        self.g2E[1,1].submatrix11,
                        self.g3E[0,0].submatrix11,
                        self.g3E[1,1].submatrix11,
                        self.gammaLE.submatrix01,
                        self.gammaLE.submatrix10,
                        self.deltaGammaE.submatrix01,
                        self.deltaGammaE.submatrix10,
                        self.yLE.submatrix01,
                        self.yLE.submatrix10,
                        self.currentE[0,0].submatrix01,
                        self.currentE[0,0].submatrix10,
                        self.currentE[1,1].submatrix01,
                        self.currentE[1,1].submatrix10,
                        self.g2E[0,1].values,
                        self.g2E[1,0].values,
                        self.g3E[0,1].values,
                        self.g3E[1,0].values,
                        self.currentE[0,1].values,
                        self.currentE[1,0].values,
                    ]
            if self.include_Ga:
                try:
                    all_data += [
                            self.ga_scalarE.submatrix00,
                            self.ga_scalarE.submatrix11,
                            ]
                except AttributeError:
                    all_data += [
                            self.gaE[0,0].submatrix00,
                            self.gaE[1,1].submatrix00,
                            self.gaE[0,0].submatrix11,
                            self.gaE[1,1].submatrix11,
                            self.gaE[0,1].values,
                            self.gaE[1,0].values,
                            ]
            return np.concatenate(all_data, axis=None)
        elif self.compact == 2:
            if settings.CHECK_SYMMETRIES:
                assert self.gammaE.symmetry == -1
                assert self.gammaE.submatrix01 is None
                assert self.gammaE.submatrix10 is None
                assert self.zE.symmetry == -1
                assert self.zE.submatrix01 is None
                assert self.zE.submatrix10 is None
                assert self.deltaGammaLE.symmetry == -1
                assert self.deltaGammaLE.submatrix01 is None
                assert self.deltaGammaLE.submatrix10 is None
                assert self.gammaLE.symmetry == -1
                assert self.gammaLE.submatrix00 is None
                assert self.gammaLE.submatrix11 is None
                assert self.deltaGammaE.symmetry == -1
                assert self.deltaGammaE.submatrix00 is None
                assert self.deltaGammaE.submatrix11 is None
                assert self.yLE.symmetry == 1
                assert self.yLE.submatrix00 is None
                assert self.yLE.submatrix11 is None
                assert self.g2E.symmetry == -1
                assert self.g3E.symmetry == 1
                assert self.currentE.symmetry == 1
                assert self.g2E[0,0].submatrix01 is None
                assert self.g3E[0,0].submatrix01 is None
                assert self.g2E[0,0].submatrix10 is None
                assert self.g3E[0,0].submatrix10 is None
                assert self.currentE[0,0].submatrix00 is None
                assert self.currentE[0,0].submatrix11 is None
            if self.yLE.submatrix01 is None:
                dummy = np.zeros(self.nmax*(self.nmax+1), dtype=np.complex128)
                self.yLE.submatrix01 = dummy
                self.yLE.submatrix10 = dummy
                if self.deltaGammaE.submatrix01 is None:
                    self.deltaGammaE.submatrix01 = dummy
                    self.deltaGammaE.submatrix10 = dummy
                if self.currentE[0,0].submatrix01 is None:
                    self.currentE[0,0].submatrix01 = dummy
                    self.currentE[0,0].submatrix10 = dummy
            pack_flat = lambda x, sym: (x.reshape(-1)[:(x.size+1)//2] + sym*x.reshape(-1)[:x.size//2-1:-1].conjugate())/2
            all_data = [
                        pack_flat(self.gammaE.submatrix00, -1),
                        pack_flat(self.zE.submatrix00, -1),
                        pack_flat(self.deltaGammaLE.submatrix00, -1),
                        pack_flat(self.g2E[0,0].submatrix00, -1),
                        pack_flat(self.g3E[0,0].submatrix00, 1),
                        pack_flat(self.gammaE.submatrix11, -1),
                        pack_flat(self.zE.submatrix11, -1),
                        pack_flat(self.deltaGammaLE.submatrix11, -1),
                        pack_flat(self.g2E[0,0].submatrix11, -1),
                        pack_flat(self.g3E[0,0].submatrix11, 1),
                        pack_flat(self.gammaLE.submatrix01, -1),
                        pack_flat(self.gammaLE.submatrix10, -1),
                        pack_flat(self.deltaGammaE.submatrix01, -1),
                        pack_flat(self.deltaGammaE.submatrix10, -1),
                        pack_flat(self.yLE.submatrix01, 1),
                        pack_flat(self.yLE.submatrix10, 1),
                        pack_flat(self.currentE[0,0].submatrix01, 1),
                        pack_flat(self.currentE[0,0].submatrix10, 1),
                        self.g2E[0,1].values,
                        self.g3E[0,1].values,
                        self.currentE[0,1].values,
                    ]
            if self.include_Ga:
                assert self.ga_scalarE.submatrix01 is None
                assert self.ga_scalarE.submatrix10 is None
                assert np.allclose(self.ga_scalarE.submatrix00.flat, self.ga_scalarE.submatrix00.flat[::-1].conjugate())
                assert np.allclose(self.ga_scalarE.submatrix11.flat, self.ga_scalarE.submatrix11.flat[::-1].conjugate())
                all_data += [
                        pack_flat(self.ga_scalarE.submatrix00, 1),
                        pack_flat(self.ga_scalarE.submatrix11, 1),
                        ]
            return np.concatenate(all_data, axis=None)


    def packFlattenedDerivativesMinimal(self):
        """
        Pack Floquet matrices representing derivatives in one flattened
        (1d) array for the solver.

        Order of flattened_values:
        Γ, Z, δΓ, *G2, *Ga
        """
        if self.compact == 0:
            all_data = [
                        self.gammaE.values,
                        self.zE.values,
                        *(g.values for g in self.g2E.data.flat),
                    ]
            if self.include_Ga:
                try:
                    all_data.append(self.ga_scalarE.values)
                except AttributeError:
                    all_data += [*(g.values for g in self.gaE.data.flat)]
            return np.concatenate(all_data, axis=None)
        elif self.compact == 1:
            if settings.CHECK_SYMMETRIES:
                assert self.gammaE.submatrix01 is None
                assert self.gammaE.submatrix10 is None
                assert self.zE.submatrix01 is None
                assert self.zE.submatrix10 is None
                assert self.g2E[0,0].submatrix01 is None
                assert self.g2E[1,1].submatrix01 is None
                assert self.g2E[0,0].submatrix10 is None
                assert self.g2E[1,1].submatrix10 is None
            all_data = [
                        self.gammaE.submatrix00,
                        self.zE.submatrix00,
                        self.g2E[0,0].submatrix00,
                        self.g2E[1,1].submatrix00,
                        self.gammaE.submatrix11,
                        self.zE.submatrix11,
                        self.g2E[0,0].submatrix11,
                        self.g2E[1,1].submatrix11,
                        self.g2E[0,1].values,
                        self.g2E[1,0].values,
                    ]
            if self.include_Ga:
                try:
                    all_data += [
                            self.ga_scalarE.submatrix00,
                            self.ga_scalarE.submatrix11,
                            ]
                except AttributeError:
                    all_data += [
                            self.gaE[0,0].submatrix00,
                            self.gaE[1,1].submatrix00,
                            self.gaE[0,0].submatrix11,
                            self.gaE[1,1].submatrix11,
                            self.gaE[0,1].values,
                            self.gaE[1,0].values,
                            ]
            return np.concatenate(all_data, axis=None)
        elif self.compact == 2:
            if self.include_Ga:
                raise NotImplementedError
            if settings.CHECK_SYMMETRIES:
                assert self.gammaE.symmetry == -1
                assert self.gammaE.submatrix01 is None
                assert self.gammaE.submatrix10 is None
                assert self.zE.symmetry == -1
                assert self.zE.submatrix01 is None
                assert self.zE.submatrix10 is None
                assert self.g2E.symmetry == -1
                assert self.g2E[0,0].submatrix01 is None
                assert self.g2E[0,0].submatrix10 is None
            pack_flat = lambda x, sym: (x.reshape(-1)[:(x.size+1)//2] + sym*x.reshape(-1)[:x.size//2-1:-1].conjugate())/2
            return np.concatenate((
                        pack_flat(self.gammaE.submatrix00, -1),
                        pack_flat(self.zE.submatrix00, -1),
                        pack_flat(self.g2E[0,0].submatrix00, -1),
                        pack_flat(self.gammaE.submatrix11, -1),
                        pack_flat(self.zE.submatrix11, -1),
                        pack_flat(self.g2E[0,0].submatrix11, -1),
                        self.g2E[0,1].values,
                ), axis=None)


    def packFlattenedValues(self):
        """
        Translate between 1d array used by the solver and Floquet matrices
        used in RG equations. Collect all Floquet matrices in one flattened
        (1d) array.

        Order of flattened_values:
        Γ, Z, δΓ, *G2, *G3, *IL, δΓL, ΓL, YL, [Ga]
        """
        if self.compact == 0:
            all_data = [
                        self.gamma.values,
                        self.z.values,
                        self.deltaGamma.values,
                        *(g.values for g in self.g2.data.flat),
                        *(g.values for g in self.g3.data.flat),
                        *(i.values for i in self.current.data.flat),
                        self.deltaGammaL.values,
                        self.gammaL.values,
                        self.yL.values,
                    ]
            if self.include_Ga:
                try:
                    all_data.append(self.ga_scalar.values)
                except AttributeError:
                    all_data += [*(g.values for g in self.ga.data.flat)]
            return np.concatenate(all_data, axis=None)
        elif self.compact == 1:
            if settings.CHECK_SYMMETRIES:
                assert self.gamma.submatrix01 is None
                assert self.gamma.submatrix10 is None
                assert self.z.submatrix01 is None
                assert self.z.submatrix10 is None
                assert self.deltaGammaL.submatrix01 is None
                assert self.deltaGammaL.submatrix10 is None
                assert self.gammaL.submatrix00 is None
                assert self.gammaL.submatrix11 is None
                assert self.deltaGamma.submatrix00 is None
                assert self.deltaGamma.submatrix11 is None
                assert self.yL.submatrix00 is None
                assert self.yL.submatrix11 is None
                assert self.g2[0,0].submatrix01 is None
                assert self.g2[1,1].submatrix01 is None
                assert self.g3[0,0].submatrix01 is None
                assert self.g3[1,1].submatrix01 is None
                assert self.g2[0,0].submatrix10 is None
                assert self.g2[1,1].submatrix10 is None
                assert self.g3[0,0].submatrix10 is None
                assert self.g3[1,1].submatrix10 is None
                assert self.current[0,0].submatrix00 is None
                assert self.current[0,0].submatrix11 is None
                assert self.current[1,1].submatrix00 is None
                assert self.current[1,1].submatrix11 is None
            all_data = [
                        self.gamma.submatrix00,
                        self.z.submatrix00,
                        self.deltaGammaL.submatrix00,
                        self.g2[0,0].submatrix00,
                        self.g2[1,1].submatrix00,
                        self.g3[0,0].submatrix00,
                        self.g3[1,1].submatrix00,
                        self.gamma.submatrix11,
                        self.z.submatrix11,
                        self.deltaGammaL.submatrix11,
                        self.g2[0,0].submatrix11,
                        self.g2[1,1].submatrix11,
                        self.g3[0,0].submatrix11,
                        self.g3[1,1].submatrix11,
                        self.gammaL.submatrix01,
                        self.gammaL.submatrix10,
                        self.deltaGamma.submatrix01,
                        self.deltaGamma.submatrix10,
                        self.yL.submatrix01,
                        self.yL.submatrix10,
                        self.current[0,0].submatrix01,
                        self.current[0,0].submatrix10,
                        self.current[1,1].submatrix01,
                        self.current[1,1].submatrix10,
                        self.g2[0,1].values,
                        self.g2[1,0].values,
                        self.g3[0,1].values,
                        self.g3[1,0].values,
                        self.current[0,1].values,
                        self.current[1,0].values,
                    ]
            if self.include_Ga:
                try:
                    all_data += [
                            self.ga_scalar.submatrix00,
                            self.ga_scalar.submatrix11,
                            ]
                except AttributeError:
                    all_data += [
                            self.ga[0,0].submatrix00,
                            self.ga[1,1].submatrix00,
                            self.ga[0,0].submatrix11,
                            self.ga[1,1].submatrix11,
                            self.ga[0,1].values,
                            self.ga[1,0].values,
                            ]
            return np.concatenate(all_data, axis=None)
        elif self.compact == 2:
            if settings.CHECK_SYMMETRIES:
                assert self.gamma.symmetry == 1
                assert self.gamma.submatrix01 is None
                assert self.gamma.submatrix10 is None
                assert self.z.symmetry == 1
                assert self.z.submatrix01 is None
                assert self.z.submatrix10 is None
                assert self.deltaGammaL.symmetry == 1
                assert self.deltaGammaL.submatrix01 is None
                assert self.deltaGammaL.submatrix10 is None
                assert self.gammaL.symmetry == 1
                assert self.gammaL.submatrix00 is None
                assert self.gammaL.submatrix11 is None
                assert self.deltaGamma.symmetry == 1
                assert self.deltaGamma.submatrix00 is None
                assert self.deltaGamma.submatrix11 is None
                assert self.yL.symmetry == -1
                assert self.yL.submatrix00 is None
                assert self.yL.submatrix11 is None
                assert self.g2.symmetry == 1
                assert self.g3.symmetry == -1
                assert self.current.symmetry == -1
                assert self.g2[0,0].submatrix01 is None
                assert self.g3[0,0].submatrix01 is None
                assert self.g2[0,0].submatrix10 is None
                assert self.g3[0,0].submatrix10 is None
                assert self.current[0,0].submatrix00 is None
                assert self.current[0,0].submatrix11 is None
            pack_flat = lambda x, sym: (x.reshape(-1)[:(x.size+1)//2] + sym*x.reshape(-1)[:x.size//2-1:-1].conjugate())/2
            all_data = [
                        pack_flat(self.gamma.submatrix00, 1),
                        pack_flat(self.z.submatrix00, 1),
                        pack_flat(self.deltaGammaL.submatrix00, 1),
                        pack_flat(self.g2[0,0].submatrix00, 1),
                        pack_flat(self.g3[0,0].submatrix00, -1),
                        pack_flat(self.gamma.submatrix11, 1),
                        pack_flat(self.z.submatrix11, 1),
                        pack_flat(self.deltaGammaL.submatrix11, 1),
                        pack_flat(self.g2[0,0].submatrix11, 1),
                        pack_flat(self.g3[0,0].submatrix11, -1),
                        pack_flat(self.gammaL.submatrix01, 1),
                        pack_flat(self.gammaL.submatrix10, 1),
                        pack_flat(self.deltaGamma.submatrix01, 1),
                        pack_flat(self.deltaGamma.submatrix10, 1),
                        pack_flat(self.yL.submatrix01, -1),
                        pack_flat(self.yL.submatrix10, -1),
                        pack_flat(self.current[0,0].submatrix01, -1),
                        pack_flat(self.current[0,0].submatrix10, -1),
                        self.g2[0,1].values,
                        self.g3[0,1].values,
                        self.current[0,1].values,
                    ]
            if self.include_Ga:
                if settings.CHECK_SYMMETRIES:
                    assert self.ga_scalar.symmetry == -1
                    assert self.ga_scalar.submatrix01 is None
                    assert self.ga_scalar.submatrix10 is None
                all_data += [
                        pack_flat(self.ga_scalar.submatrix00, -1),
                        pack_flat(self.ga_scalar.submatrix11, -1),
                        ]
            return np.concatenate(all_data, axis=None)


    def packFlattenedValuesMinimal(self):
        """
        Translate between 1d array used by the solver and Floquet matrices
        used in RG equations. Collect all Floquet matrices in one flattened
        (1d) array.

        Order of flattened_values:
        Γ, Z, δΓ, *G2, *G3, *IL, δΓL, ΓL, YL
        """
        if self.compact == 0:
            all_data = [
                        self.gamma.values,
                        self.z.values,
                        *(g.values for g in self.g2.data.flat),
                    ]
            if self.include_Ga:
                try:
                    all_data.append(self.ga_scalar.values)
                except AttributeError:
                    all_data += [*(g.values for g in self.ga.data.flat)]
            return np.concatenate(all_data, axis=None)
        elif self.compact == 1:
            if settings.CHECK_SYMMETRIES:
                assert self.gamma.submatrix01 is None
                assert self.gamma.submatrix10 is None
                assert self.z.submatrix01 is None
                assert self.z.submatrix10 is None
                assert self.g2[0,0].submatrix01 is None
                assert self.g2[1,1].submatrix01 is None
                assert self.g2[0,0].submatrix10 is None
                assert self.g2[1,1].submatrix10 is None
            all_data = [
                        self.gamma.submatrix00,
                        self.z.submatrix00,
                        self.g2[0,0].submatrix00,
                        self.g2[1,1].submatrix00,
                        self.gamma.submatrix11,
                        self.z.submatrix11,
                        self.g2[0,0].submatrix11,
                        self.g2[1,1].submatrix11,
                        self.g2[0,1].values,
                        self.g2[1,0].values,
                    ]
            if self.include_Ga:
                try:
                    all_data += [
                            self.ga_scalar.submatrix00,
                            self.ga_scalar.submatrix11,
                            ]
                except AttributeError:
                    all_data += [
                            self.ga[0,0].submatrix00,
                            self.ga[1,1].submatrix00,
                            self.ga[0,0].submatrix11,
                            self.ga[1,1].submatrix11,
                            self.ga[0,1].values,
                            self.ga[1,0].values,
                            ]
            return np.concatenate(all_data, axis=None)
        elif self.compact == 2:
            if self.include_Ga:
                raise NotImplementedError
            if settings.CHECK_SYMMETRIES:
                assert self.gamma.symmetry == 1
                assert self.gamma.submatrix01 is None
                assert self.gamma.submatrix10 is None
                assert self.z.symmetry == 1
                assert self.z.submatrix01 is None
                assert self.z.submatrix10 is None
                assert self.g2.symmetry == 1
                assert self.g2[0,0].submatrix01 is None
                assert self.g2[0,0].submatrix10 is None
            pack_flat = lambda x, sym: (x.reshape(-1)[:(x.size+1)//2] + sym*x.reshape(-1)[:x.size//2-1:-1].conjugate())/2
            return np.concatenate((
                        pack_flat(self.gamma.submatrix00, 1),
                        pack_flat(self.z.submatrix00, 1),
                        pack_flat(self.g2[0,0].submatrix00, 1),
                        pack_flat(self.gamma.submatrix11, 1),
                        pack_flat(self.z.submatrix11, 1),
                        pack_flat(self.g2[0,0].submatrix11, 1),
                        self.g2[0,1].values,
                ), axis=None)


    def hash(self):
        data = self.packFlattenedValues()
        data.flags["WRITEABLE"] = False
        return hashlib.sha1(data.data).hexdigest()


    def odeFunctionIm(self, imenergy, flattened_values):
        """
        ODE as given to the solver for solving the RG equations along the
        imaginary axis. Given a flattened array containing all Floquet
        matrices, evaluate the RG equations and return a flattened array of
        all derivatives.
        imenergy = Im(E) is used as function argument since the solver cannot
        handle complex flow parameters.
        """
        try:
            if self.save_filename and self.save_iterations > 0 and self.iterations % self.save_iterations == 0:
                try:
                    self.save_compact()
                except:
                    settings.logger.exception("Failed to save intermediate result:")
        except AttributeError:
            pass
        except:
            settings.logger.exception("Failed trying to save intermediate result:")

        self.global_properties.energy = self.energy.real + 1j*imenergy

        self.unpackFlattenedValues(flattened_values)

        # Evaluate RG equations
        try:
            self.updateRGequations()
        except KeyboardInterrupt as error:
            settings.logger.critical('Interrupted at Im(E) = %e'%imenergy)
            try:
                self.ir_cutoff = max(self.ir_cutoff, imenergy)
            except AttributeError:
                self.ir_cutoff = imenergy
            raise error
        except Exception as error:
            settings.logger.exception('Unhandled error at Im(E) = %e'%imenergy)
            try:
                self.ir_cutoff = max(self.ir_cutoff, imenergy)
            except AttributeError:
                self.ir_cutoff = imenergy
            raise error

        # Pack values
        # use  d/d(Im E) = i d/dE
        return 1j*self.packFlattenedDerivatives()


    def odeFunctionImMinimal(self, imenergy, flattened_values):
        """
        ODE as given to the solver for solving the RG equations along the
        imaginary axis. Given a flattened array containing all Floquet
        matrices, evaluate the RG equations and return a flattened array of
        all derivatives.
        imenergy = Im(E) is used as function argument since the solver cannot
        handle complex flow parameters.
        """
        self.global_properties.energy = self.energy.real + 1j*imenergy
        self.unpackFlattenedValuesMinimal(flattened_values)

        # Evaluate RG equations
        try:
            self.updateRGequationsMinimal()
        except KeyboardInterrupt as error:
            settings.logger.critical('Interrupted at Im(E) = %e'%imenergy)
            try:
                self.ir_cutoff = max(self.ir_cutoff, imenergy)
            except AttributeError:
                self.ir_cutoff = imenergy
            raise error
        except Exception as error:
            settings.logger.exception('Unhandled error at Im(E) = %e'%imenergy)
            try:
                self.ir_cutoff = max(self.ir_cutoff, imenergy)
            except AttributeError:
                self.ir_cutoff = imenergy
            raise error
        # Pack values
        # use  d/d(Im E) = i d/dE
        return 1j*self.packFlattenedDerivativesMinimal()


    def odeFunctionRe(self, reenergy, flattened_values):
        """
        ODE as given to the solver for solving the RG equations along the
        real axis. Given a flattened array containing all Floquet matrices,
        evaluate the RG equations and return a flattened array of all
        derivatives.
        """
        if self.save_filename and self.save_iterations > 0 and self.iterations % self.save_iterations == 0:
            try:
                self.save_compact()
            except:
                settings.logger.exception('Failed to save intermediate result:')

        self.global_properties.energy = reenergy + 1j*self.energy.imag

        self.unpackFlattenedValues(flattened_values)

        # Evaluate RG equations
        try:
            self.updateRGequations()
        except KeyboardInterrupt as error:
            settings.logger.critical('Interrupted at Re(E) = %g, Im(E) = %g'%(reenergy, energy0.imag))
            raise error
        except:
            settings.logger.exception('Unhandled error at Re(E) = %g, Im(E) = %g'%(reenergy, energy0.imag))
            raise error

        # Pack values
        return self.packFlattenedDerivatives()


    def solveOdeIm(self, eiminit, eimfinal, init_values=None, only_final=False, **solveopts):
        """
        Solve the RG equations along the imaginary axis, starting from
        E = Ereal + eiminit*1j and ending at E = Ereal + eimfinal*1j
        where Ereal is the real part of the current energy.

        Other arguments:
        init_values : flattened array of initial values, by default taken from
                self.packFlattenedValues()
        only_final : only save final result, do not save the RG flow.
                This saves memory.
        **solveopts : arguments that are directly passed on to the solver. Most
                relevant are rtol and atol.
        """
        assert np.allclose(self.energy.imag, eiminit)
        if init_values is None:
            init_values = self.packFlattenedValues()
        output = solve_ivp(
                self.odeFunctionIm,
                (eiminit, eimfinal),
                init_values,
                t_eval = solveopts.pop("t_eval", ((eimfinal,) if only_final else None)),
                **solveopts
            )
        return output


    def solveOdeRe(self, reEinit, reEfinal, init_values=None, only_final=False, **solveopts):
        """
        Solve the RG equations along the real axis, starting from
        E = reEinit + 1j*Eimag and ending at E = reEfinal + 1j*Eimag.
        where Eimag is the imaginary part of the current energy.

        Other arguments:
        init_values : flattened array of initial values, by default taken from
                self.packFlattenedValues()
        only_final : only save final result, do not save the RG flow.
                This saves memory.
        **solveopts : arguments that are directly passed on to the solver. Most
                relevant are rtol and atol.
        """
        assert abs(self.energy.real - reEinit) < 1e-8
        if init_values is None:
            init_values = self.packFlattenedValues()
        output = solve_ivp(
                self.odeFunctionRe,
                (reEinit, reEfinal),
                init_values,
                t_eval = solveopts.pop("t_eval", ((eimfinal,) if only_final else None)),
                **solveopts
            )
        return output


    def findPole(self, **solveopts):
        """
        Should be called after RG flow has run until E=0 and
        values have been saved.
        """
        assert abs(self.energy.real) < 1e-12
        init_values = self.packFlattenedValuesMinimal()
        raise NotImplementedError
        idx = 0 # TODO
        event = lambda t, y: t + y[idx] - 1e-3
        output = solve_ivp(
                self.odeFunctionImMinimal,
                (self.energy.imag, -1e6),
                init_values,
                events = event,
                t_eval = (),
                **solveopts
            )
        self.pole = -output.t[-1]
        return self.pole, output


    def save_compact(self, values, compressed=False):
        """
        Automatically save current state of the RG flow in the most compact
        form. The file name will be
            self.save_filename % self.iterations
        or
            self.save_filename.format(self.iterations).
        In a computationally expensive RG flow this allows saving intermediate
        steps of the RG flow.
        """
        try:
            filename = self.save_filename%self.iterations
        except TypeError:
            filename = self.save_filename.format(self.iterations)
        (np.savez_compressed if compressed else np.savez)(
                filename,
                values = values,
                energy = self.energy,
                compact = self.compact,
                )


    def load_compact(self, filename):
        """
        Load a file that was created with Kondo.save_compact. This overwrites
        the current state of self with the values given in the file.
        """
        data = np.load(filename)
        assert data['compact'] == self.compact
        self.unpackFlattenedValues(data['values'])
        self.global_properties.energy = data['energy']
