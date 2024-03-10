# Copyright 2021 Valentin Bruch <valentin.bruch@rwth-aachen.de>
# License: MIT
"""
Kondo FRTRG, module defining RG objects

Module defining classes GlobalProperties, RGobj, and RGfunction for objects
appearing in RG equations. A GlobalProperties object is shared by instances
of RGobj to ensure they are compatible. RGfunction defines a Floquet matrix.

See also: reservoirmatrix.py, compact_rtrg.py
"""

import numpy as np
from numbers import Number
from scipy.interpolate import interp1d
from . import settings

# rtrg_c contains functions written in C to speed up the calculation.
# In principle these funtions are also available using CUBLAS to boost
# the matrix multiplication. However, it is usually not efficient to use
# the CUBLAS version.
if settings.USE_CUBLAS:
    try:
        from . import rtrg_cublas as rtrg_c
    except:
        settings.logger.warning(
            "Failed to load rtrg_cublas, falling back to rtrg_c", exc_info=True
        )
        from . import rtrg_c
else:
    from . import rtrg_c


class GlobalRGproperties:
    """
    Shared properties container object for RGfunction.
    The properties stored in this object must be shared by all RGfunctions
    which are used together in the same calculation. Usually all RGfunctions
    own a reference to the same GlobalRGproperties object, which makes it
    simple to adopt the energies of all these RGfunctions and to check whether
    different RGfunctions are compatible.
    """

    DEFAULTS = dict(
        nmax=0,
        padding=0,
        voltage_branches=0,
        resonant_dc_shift=0,
        energy=0j,
        symmetric=True,
        mu=None,
        vdc=0,
        clear_corners=0,
    )

    def __init__(self, **kwargs):
        """
        Shared object containing common information associated with
        Floquet matrices that are used within the same Kondo system.

        For each run of solving a physical system at fixed physical
        and numerical parameters, these parameters are collected in
        this object, which is shared by all Floquet matrices.
        Furthermore, this object contains the Laplace variable E
        (energy) which is the flow parameter.

        Parameters
        ----------
        energy : complex
            RG flow parameter, Laplace variable with physical meaning
            of energy
        omega : float
            frequency :math:`\\Omega` of the driving
        nmax : int
            the shape of the Floquet matrices is given by
            (2*nmax+1, 2*nmax+1)
        voltage_branches : int, default=0
            copies of the RG variables at energies shifted by
            :math:`n\\cdot\\mu_{LR}` or :math:`n\\cdot V_{avg}` where
            n=-voltage_branches...voltage_branches
        vdc : int, default=0
            :math:`V_{avg}`, average voltage.
            Either `vdc` or `mu` must be provided.
        mu : array, default=0
            :math:`\\mu_{LR}`, Floquet matrix representing chemical
            potential difference. Either `vdc` or `mu` must be
            provided.
        """
        settings.logger.debug("Created new GlobalRGproperties object")
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        """
        Return property of take value from DEFAULTS if property is not set.

        Parameter
        ---------
        name : str
            name of the attribute

        Returns
        -------
        attribute, using default value if it is not set
        """
        try:
            return self.DEFAULTS[name]
        except KeyError:
            raise AttributeError()

    def shape(self):
        """
        Shape of RGfunction.values for all RGfunctions with these global
        properties.
        """
        floquet_dim = 2 * self.__dict__["nmax"] + 1
        if self.__dict__["voltage_branches"]:
            return (2 * self.__dict__["voltage_branches"] + 1, floquet_dim, floquet_dim)
        else:
            return (floquet_dim, floquet_dim)

    def copy(self):
        """
        Create a copy of self. It is assumed that all attributes of self are
        implicitly copied on assignment.

        Returns
        -------
        copy of self
        """
        return GlobalRGproperties(**self.__dict__)


class RGobj:
    def __init__(self, global_properties: GlobalRGproperties, symmetry: int = 0):
        """
        Object owning a copy of :class:`GlobalRGproperties` that has
        a symmetry attribute.

        Parameters
        ----------
        global_properties : :class:`GlobalRGProperties` object
            shared properties containing information about the
            physical system and the numerical parameters
        symmetry : {0, 1, -1}
            symmetry of the object under Floquet transposition
            (taking the adjoint and reverting the indices)
        """
        self.global_properties = global_properties
        self.symmetry = symmetry

    def __getattr__(self, name):
        """
        Convenience function for easily accessing global properties
        like attributes of self

        Parameters
        ----------
        name : str
            name of an attribute

        Returns
        -------
        attribute
            taken from global properties
        """
        return getattr(self.global_properties, name)


class RGfunction(RGobj):
    """
    Matrix X_{nm}(E) = X_{n-m}(E+mΩ) with n,m = -nmax...nmax

    self.values:
    This contains an array of the shapes
    (2*voltage_branches+1, 2*nmax+1, 2*nmax+1) or (2*nmax+1, 2*nmax+1) as values.

    self.voltage_shifts:
    energy shifts (in units of DC voltage) which should be included in all
    RGfunctions standing on the right of self.
    In a product the voltage_shifts of both RGfunctions are added.
    This is just for book keeping in products of RGfunctions.

    self.global_properties:
    Additionally, some properties (energy, omega, voltage, nmax, ...)
    are stored in the shared object global_properties.

    self.symmetry:
    Valid values are 0, -1, +1. If non-zero, it states the symmetry
    X[::-1,::-1,::-1] == symmetry * X.conjugate() for this Floquet matrix.
    """

    def __init__(
        self, global_properties, values=None, voltage_shifts=0, symmetry=0, **kwargs
    ):
        """
        Floquet matrix

        See the literature for a definition of Floquet matrices and
        their properties. While in analytic calculations we usually
        consider Floquet matrices of infinite size and indices running
        from :math:`-\\infty` to :math:`\\infty`, here we represent
        Floquet matricse of finite size by numpy arrays.
        The arrays represent Floquet matrices with indices
        `-nmax...nmax`, but in the numpy array these are shifted by
        `nmax` to `0...2*nmax`.

        If `global_properties.voltage_branches > 0`, the
        voltage-shifted copies are also included such that the shape
        of the numpy array internally representing this Floquet matrix
        is `(2*voltage_branches+1, 2*nmax+1, 2*nmax+1)`.

        Parameters
        ----------
        global_properties : :class:`GlobalRGProperties` object
            shared properties of the system of interest and numerical
            parameters
        values : array
            numpy array representing the Floquet matrix. If
            `voltage_branches == 0`, this must have the shape
            `(2*nmax+1, 2*nmax+1)` (with variables taken from
            `global_properties`). Otherwise, it must have the shape
            `(2*voltage_branches+1, 2*nmax+1, 2*nmax+1)`.
        voltage_shifts : int, default=0
            shift of the diagonal that can account for a constant bias
            voltage if the time dependence of the driving is included
            in the system-reservoir coupling by a unitary
            transformation of the Hamiltonian
        symmetry : int, default=0
            symmetry of the Floquet matrix under the Floquet
            conjugation as defined in :meth:`~floquetConjugate`
        **kwargs : dict
            attributes added to self
        """
        super().__init__(global_properties, symmetry)
        self.energy_shifted_copies = {}
        self.voltage_shifts = voltage_shifts
        for key, value in kwargs.items():
            setattr(self, key, value)
        if values is None:
            # should be implemented by a class inheriting from RGfunction:
            self._values = self.initial()
        elif type(values) == str and values == "identity":
            self.symmetry = 1
            if self.voltage_branches:
                self._values = np.broadcast_to(
                    np.identity(2 * self.nmax + 1, dtype=np.complex128).reshape(
                        1, 2 * self.nmax + 1, 2 * self.nmax + 1
                    ),
                    self.shape(),
                )
            else:
                self._values = np.identity(2 * self.nmax + 1, dtype=np.complex128).T
        else:
            self._values = np.asarray(values, dtype=np.complex128)
            assert self._values.shape[-1] == self._values.shape[-2] == 2 * self.nmax + 1

    @property
    def values(self):
        """Numpy array representing this matrix"""
        return self._values

    @values.setter
    def values(self, values):
        assert self._values.shape[-1] == self._values.shape[-2] == 2 * self.nmax + 1
        self._values = values
        self.energy_shifted_copies.clear()

    def copy(self):
        """
        Create a copy of self. Values (the array representing the
        Floquet matrix) are copied, the global_properties are shared
        as a reference.

        Returns
        -------
        copy of self
        """
        return RGfunction(
            self.global_properties,
            self._values.copy(),
            self.voltage_shifts,
            self.symmetry,
        )

    def floquetConjugate(self):
        """
        For a Floquet matrix A(E)_{nm} this returns the C-transform
        .. math:: C A(E)_{nm} C = A(-E*)_{-n,-m}
        with the superoperator C defined by
        .. math:: C x := x^\dag.
        This uses the symmetry of self if self has a symemtry. If this
        C-transform leaves self invariant, this function will return a
        copy of self, but never a reference to self.

        This can only be evaluated if the energy of self lies on the
        imaginary axis.
        """
        if self.symmetry == 1:
            return self.copy()
        elif self.symmetry == -1:
            return -self
        if self._values.ndim == 2:
            assert abs(self.energy.real) < 1e-12
            return RGfunction(
                self.global_properties,
                np.conjugate(self._values[::-1, ::-1]),
                self.voltage_shifts,
            )
        elif self._values.ndim == 3:
            assert abs(self.energy.real) < 1e-12
            return RGfunction(
                self.global_properties,
                np.conjugate(self._values[::-1, ::-1, ::-1]),
                self.voltage_shifts,
            )
        else:
            raise NotImplementedError()

    def __matmul__(self, other):
        """
        Convolution (or product in Floquet space) of two RG functions.
        Other must be of type RGfunction.

        Note: This is only approximately associative, as long the function
        converges to 0 for :math:`|n|` of order of nmax.
        """
        if isinstance(other, SymRGfunction):
            return NotImplemented
        if not isinstance(other, RGfunction):
            return NotImplemented
        assert self.global_properties is other.global_properties
        if self._values.ndim == 2 and other._values.ndim == 3:
            symmetry = self.symmetry * other.symmetry * (self.voltage_shifts == 0)
            vb = other._values.shape[0] // 2
            matrix = rtrg_c.multiply_extended(
                other._values[vb + self.voltage_shifts].T,
                self._values.T,
                self.padding,
                symmetry,
                self.clear_corners,
            ).T
        else:
            assert self._values.shape == other._values.shape
            right = other.shift_energies(self.voltage_shifts)
            symmetry = self.symmetry * other.symmetry * (self.voltage_shifts == 0)
            matrix = rtrg_c.multiply_extended(
                right._values.T,
                self._values.T,
                self.padding,
                symmetry * (self._values.ndim == 2),
                self.clear_corners,
            ).T
        res = RGfunction(
            self.global_properties,
            matrix,
            self.voltage_shifts + other.voltage_shifts,
            symmetry,
        )
        return res

    def __imatmul__(self, other):
        if isinstance(other, SymRGfunction):
            return NotImplemented
        if not isinstance(other, RGfunction):
            return NotImplemented
        assert self.global_properties is other.global_properties
        self.energy_shifted_copies.clear()
        if self._values.ndim == 2 and other._values.ndim == 3:
            symmetry = self.symmetry * other.symmetry * (self.voltage_shifts == 0)
            vb = other._values.shape[0] // 2
            matrix = rtrg_c.multiply_extended(
                other._values[vb + self.voltage_shifts].T,
                self._values.T,
                self.padding,
                symmetry,
                self.clear_corners,
            ).T
        else:
            assert self._values.shape == other._values.shape
            right = other.shift_energies(self.voltage_shifts)
            self.symmetry *= other.symmetry
            self._values = rtrg_c.multiply_extended(
                right._values.T,
                self._values.T,
                self.padding,
                self.symmetry * (self._values.ndim == 2),
                self.clear_corners,
                OVERWRITE_LEFT,
            ).T
        self.voltage_shifts += other.voltage_shifts
        return self

    def __add__(self, other):
        """
        Add other to self. If other is a scalar or a scalar function of energy
        represented by an array of values at self.energies, this treats other
        as an identity (or diagonal) Floquet matrix.
        Other must be a scalar or array of same shape as self.energies or an
        RGfunction of the same shape and energies as self.
        """
        if isinstance(other, RGfunction):
            assert self.global_properties is other.global_properties
            assert self.voltage_shifts == other.voltage_shifts
            symmetry = (self.symmetry == other.symmetry) * self.symmetry
            return RGfunction(
                self.global_properties,
                self._values + other._values,
                self.voltage_shifts,
                symmetry,
            )
        elif np.shape(other) == () or np.shape(other) == (2 * self.nmax + 1,):
            # TODO: symmetries
            # Assume that other represents a (possibly energy-dependent) scalar.
            newvalues = self._values.copy()
            newvalues[(..., *np.diag_indices(self._values.shape[-1]))] += other
            return RGfunction(self.global_properties, newvalues, self.voltage_shifts)
        else:
            raise TypeError(
                "unsupported operand types for +: RGfunction and", type(other)
            )

    def __sub__(self, other):
        if isinstance(other, RGfunction):
            assert self.global_properties is other.global_properties
            assert self.voltage_shifts == other.voltage_shifts
            symmetry = (self.symmetry == other.symmetry) * self.symmetry
            return RGfunction(
                self.global_properties,
                self._values - other._values,
                self.voltage_shifts,
                symmetry,
            )
        elif np.shape(other) == () or np.shape(other) == (2 * self.nmax + 1,):
            # TODO: symmetries
            # Assume that other represents a (possibly energy-dependent) scalar.
            newvalues = self._values.copy()
            newvalues[(..., *np.diag_indices(self._values.shape[-1]))] -= other
            return RGfunction(self.global_properties, newvalues, self.voltage_shifts)
        else:
            raise TypeError(
                "unsupported operand types for +: RGfunction and", type(other)
            )

    def __neg__(self):
        """
        Return a copy of self with inverted sign of self._values.
        """
        return RGfunction(
            self.global_properties, -self._values, self.voltage_shifts, self.symmetry
        )

    def __iadd__(self, other):
        if isinstance(other, RGfunction):
            assert self.global_properties is other.global_properties
            assert self.voltage_shifts == other.voltage_shifts
            self._values += other._values
            self.symmetry = (self.symmetry == other.symmetry) * self.symmetry
        elif np.shape(other) == () or np.shape(other) == (2 * self.nmax + 1,):
            # TODO: symmetries
            # Assume that other represents a (possibly energy-dependent) scalar.
            self._values[(..., *np.diag_indices(self._values.shape[-1]))] += other
            self.symmetry = 0
        else:
            raise TypeError(
                "unsupported operand types for +: RGfunction and", type(other)
            )
        self.energy_shifted_copies.clear()
        return self

    def __isub__(self, other):
        if isinstance(other, RGfunction):
            assert self.global_properties is other.global_properties
            assert self.voltage_shifts == other.voltage_shifts
            self._values -= other._values
            self.symmetry = (self.symmetry == other.symmetry) * self.symmetry
        elif np.shape(other) == () or np.shape(other) == (2 * self.nmax + 1,):
            # TODO: symmetries
            # Assume that other represents a (possibly energy-dependent) scalar.
            self._values[(..., *np.diag_indices(self._values.shape[-1]))] -= other
        else:
            raise TypeError(
                "unsupported operand types for +: RGfunction and", type(other)
            )
        self.energy_shifted_copies.clear()
        return self

    def __mul__(self, other):
        """
        Multiplication by scalar or RGfunction.
        If other is an RGfunction, this calculates the matrix product
        (equivalent to __matmul__). If other is a scalar, this multiplies
        self._values by other.
        """
        if isinstance(other, RGfunction):
            return self @ other
        if isinstance(other, Number):
            if other.imag == 0:
                symmetry = self.symmetry
            elif other.real == 0:
                symmetry = -self.symmetry
            else:
                symmetry = 0
            return RGfunction(
                self.global_properties,
                other * self._values,
                self.voltage_shifts,
                symmetry,
            )
        return NotImplemented

    def __imul__(self, other):
        """
        In-place multiplication by scalar or RGfunction.
        If other is an RGfunction, this calculates the matrix product
        (equivalent to __matmul__). If other is a scalar, this multiplies
        self._values by other.
        """
        if isinstance(other, RGfunction):
            self @= other
            self.energy_shifted_copies.clear()
        elif isinstance(other, Number):
            if other.imag != 0:
                if other.real == 0:
                    self.symmetry = -self.symmetry
                else:
                    self.symmetry = 0
            self._values *= other
            for copy in self.energy_shifted_copies.values():
                copy *= other
        else:
            return NotImplemented
        return self

    def __rmul__(self, other):
        """
        Reverse multiplication by scalar or RGfunction.
        If other is an RGfunction, this calculates the matrix product
        (equivalent to __matmul__). If other is a scalar, this multiplies
        self._values by other.
        """
        if isinstance(other, RGfunction):
            return other @ self
        if isinstance(other, Number):
            if other.imag == 0:
                symmetry = self.symmetry
            elif other.real == 0:
                symmetry = -self.symmetry
            else:
                symmetry = 0
            return RGfunction(
                self.global_properties,
                other * self._values,
                self.voltage_shifts,
                symmetry,
            )
        return NotImplemented

    def __truediv__(self, other):
        """
        Divide self by other, which must be a scalar.
        """
        if isinstance(other, Number):
            if other.imag == 0:
                symmetry = self.symmetry
            elif other.real == 0:
                symmetry = -self.symmetry
            else:
                symmetry = 0
            return RGfunction(
                self.global_properties,
                self._values / other,
                self.voltage_shifts,
                symmetry,
            )
        return NotImplemented

    def __itruediv__(self, other):
        """
        Divide self in-place by other, which must be a scalar.
        """
        if isinstance(other, Number):
            if other.imag != 0:
                if other.real == 0:
                    self.symmetry = -self.symmetry
                else:
                    self.symmetry = 0
            self._values /= other
            for copy in self.energy_shifted_copies.values():
                copy /= other
            return self
        return NotImplemented

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __str__(self):
        return str(self._values)

    def __repr__(self):
        return "RGfunction{ %s,\n%s }" % (self.energy, self._values.__repr__())

    def __getitem__(self, arg):
        return self._values.__getitem__(arg)

    def __eq__(self, other):
        return (
            (self.global_properties is other.global_properties)
            and self.voltage_shifts == other.voltage_shifts
            and np.allclose(self._values, other._values)
            and self.symmetry == other.symmetry
        )

    def k2lambda(self, shift_matrix=None, calculate_energy_shifts=False):
        """
        shift is usually n*zinv*mu.

        This calculates the propagator from an effective Liouvillian.
        Some of the linear systems of equation which we need to solve here are
        overdetermined. This means that we can in general only get an
        approximate solution because an exact solution does not exist.
        """
        if shift_matrix is not None:
            shift_matrix_vb = shift_matrix._values.shape[0] // 2
        if self._values.ndim == 3 and calculate_energy_shifts:
            invert_array = np.ndarray(
                (2 * self.voltage_branches + 5, 2 * self.nmax + 1, 2 * self.nmax + 1),
                dtype=np.complex128,
            )
            invert_array[2:-2] = -self._values
            for v in range(-self.voltage_branches, self.voltage_branches + 1):
                invert_array[
                    (v + 2 + self.voltage_branches, *np.diag_indices(2 * self.nmax + 1))
                ] += (
                    self.energy
                    + v * self.vdc
                    + self.omega * np.arange(-self.nmax, self.nmax + 1)
                )
                if shift_matrix is not None:
                    invert_array[v + 2 + self.voltage_branches] += shift_matrix[
                        v + shift_matrix_vb
                    ]
            dim0 = 2 * self.voltage_branches + 1
            if settings.EXTRAPOLATE_VOLTAGE:
                try:
                    interp = interp1d(
                        np.arange(dim0),
                        invert_array[2:-2],
                        "quadratic",
                        axis=0,
                        fill_value="extrapolate",
                    )
                except ValueError:
                    interp = interp1d(
                        np.arange(dim0),
                        invert_array[2:-2],
                        "linear",
                        axis=0,
                        fill_value="extrapolate",
                    )
                invert_array[-2:] = interp(dim0 + np.arange(2))
                invert_array[:2] = interp(np.arange(-2, 0))
            else:
                invert_array[-2:] = invert_array[-3]
                if shift_matrix is not None:
                    shift_matrix_vb = shift_matrix._values.shape[0] // 2
                    invert_array[-2:] += shift_matrix[
                        shift_matrix_vb + 1 : shift_matrix_vb + 3
                    ]
                    invert_array[:2] = invert_array[2]
                    invert_array[:2] += shift_matrix[
                        shift_matrix_vb - 2 : shift_matrix_vb
                    ]
            inverted = rtrg_c.invert_extended(
                invert_array.T,
                self.padding,
                round(settings.LAZY_INVERSE_FACTOR * self.padding),
            ).T
            symmetry = (
                -1 if (self.symmetry == -1 and shift_matrix.symmetry == -1) else 0
            )
            res = RGfunction(
                self.global_properties,
                inverted[2:-2],
                voltage_shifts=self.voltage_shifts,
                symmetry=symmetry,
            )
            res.energy_shifted_copies[2] = RGfunction(
                self.global_properties, inverted[4:], self.voltage_shifts, symmetry=0
            )
            res.energy_shifted_copies[1] = RGfunction(
                self.global_properties, inverted[3:-1], self.voltage_shifts, symmetry=0
            )
            res.energy_shifted_copies[-1] = RGfunction(
                self.global_properties, inverted[1:-3], self.voltage_shifts, symmetry=0
            )
            res.energy_shifted_copies[-2] = RGfunction(
                self.global_properties, inverted[:-4], self.voltage_shifts, symmetry=0
            )
            return res
        else:
            invert = -self
            if self._values.ndim == 3:
                for v in range(-self.voltage_branches, self.voltage_branches + 1):
                    invert._values[
                        (v + self.voltage_branches, *np.diag_indices(2 * self.nmax + 1))
                    ] += (
                        self.energy
                        + v * self.vdc
                        + self.omega * np.arange(-self.nmax, self.nmax + 1)
                    )
                    if shift_matrix is not None:
                        invert._values[v + self.voltage_branches] += shift_matrix[
                            v + shift_matrix_vb
                        ]
            elif self._values.ndim == 2:
                invert._values[
                    np.diag_indices(2 * self.nmax + 1)
                ] += self.energy + self.omega * np.arange(-self.nmax, self.nmax + 1)
                if shift_matrix is not None:
                    invert._values += shift_matrix[shift_matrix_vb]
            else:
                raise ValueError("Invalid RG object (shape %s)" % invert._values.shape)
            invert.symmetry = (
                -1
                if (
                    self.symmetry == -1
                    and (shift_matrix is None or shift_matrix.symmetry == -1)
                )
                else 0
            )
            return invert.inverse()

    def reduced(self, shift=0):
        """
        Remove voltage-shifted copies
        """
        if self._values.ndim == 2 and shift == 0:
            return self
        assert self._values.ndim == 3
        assert abs(shift) <= self.voltage_branches
        return RGfunction(
            self.global_properties,
            self._values[self.voltage_branches + shift],
            symmetry=self.symmetry * (shift == 0),
        )

    def reduced_to_voltage_branches(self, voltage_branches):
        if self._values.ndim == 2 or 2 * voltage_branches + 1 == self._values.shape[0]:
            return self
        assert 0 < voltage_branches < self._values.shape[0] // 2
        assert self._values.ndim == 3
        diff = self._values.shape[0] // 2 - voltage_branches
        result = RGfunction(
            self.global_properties,
            self._values[diff:-diff],
            symmetry=self.symmetry,
            voltage_shifts=self.voltage_shifts,
        )
        try:
            result.eigvals = self.eigvals[diff:-diff]
            result.reigvecs = self.reigvecs[diff:-diff]
            result.leigvecs = self.leigvecs[diff:-diff]
        except AttributeError:
            pass
        return result

    def inverse(self):
        """
        multiplicative inverse
        """
        assert self.voltage_shifts == 0
        if self.padding == 0 or settings.LAZY_INVERSE_FACTOR * self.padding < 0.5:
            res = np.linalg.inv(self._values)
        else:
            try:
                res = rtrg_c.invert_extended(
                    self._values.T,
                    self.padding,
                    round(settings.LAZY_INVERSE_FACTOR * self.padding),
                ).T
            except:
                settings.logger.exception("padded inversion failed")
                res = np.linalg.inv(self._values)
        return RGfunction(
            self.global_properties, res, self.voltage_shifts, self.symmetry
        )

    def shift_energies(self, n=0):
        # TODO: use symmetries
        """
        Shift energies by n*self.vdc. The calculated RGfunction is kept in cache.
        Assumptions:
        * On the last axis of self.energies is linear with values separated by self.vdc
        * n is an integer
        * derivative is a RGfunction or None
        * if the length of the last axis of self.energies is < 2, then derivative must not be None
        """
        if n == 0 or (self.vdc == 0 and self.mu is None):
            return self
        try:
            return self.energy_shifted_copies[n]
        except KeyError:
            assert self._values.ndim == 3
            newvalues = array_shift(self._values, n)
            self.energy_shifted_copies[n] = RGfunction(
                self.global_properties, newvalues, self.voltage_shifts
            )
            return self.energy_shifted_copies[n]

    def check_symmetry(self):
        assert self.symmetry in (-1, 0, 1)
        if self.symmetry:
            if self._values.ndim == 2:
                conjugate = self._values[::-1, ::-1].conjugate()
            elif self._values.ndim == 3:
                conjugate = self._values[::-1, ::-1, ::-1].conjugate()
            assert np.allclose(self._values, self.symmetry * conjugate)

    def diagonalize(self):
        """
        return eigenvalues, right eigenvectors, left eigenvectors
        such that
        |   self[...,i,j] = Σ reigvecs[...,i,k] @ eigenvalues[...,k] @ leigvecs[...,k,j]
        |            k
        """
        try:
            return self.eigvals, self.reigvecs, self.leigvecs
        except AttributeError:
            self.eigvals, self.reigvecs = np.linalg.eig(self.values)
            self.leigvecs = np.linalg.inv(self.reigvecs)
            return self.eigvals, self.reigvecs, self.leigvecs


def array_shift(array, n):
    newarray = np.empty_like(array)
    if settings.EXTRAPOLATE_VOLTAGE:
        try:
            interp = interp1d(
                np.arange(array.shape[0]),
                array,
                "quadratic",
                axis=0,
                fill_value="extrapolate",
            )
        except ValueError:
            interp = interp1d(
                np.arange(array.shape[0]),
                array,
                "linear",
                axis=0,
                fill_value="extrapolate",
            )
        if n > 0:
            newarray[:-n] = array[n:]
            newarray[-n:] = interp(array.shape[0] + np.arange(n))
        else:
            newarray[-n:] = array[:n]
            newarray[:-n] = interp(np.arange(n, 0))
    else:
        if n > 0:
            newarray[:-n] = array[n:]
            newarray[-n:] = array[-1:]
        else:
            newarray[-n:] = array[:n]
            newarray[:-n] = array[:1]
    return newarray


from .compact_rtrg import SymRGfunction, OVERWRITE_LEFT, OVERWRITE_BOTH, OVERWRITE_RIGHT
