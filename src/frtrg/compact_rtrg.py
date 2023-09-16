# Copyright 2021 Valentin Bruch <valentin.bruch@rwth-aachen.de>
# License: MIT
"""
Kondo FRTRG, module defining RG objects describing Floquet matrices with
special symmetry

Module defining class SymRGfunction for the special symmetric case of driving
fulfilling  V(t + T/2) = - V(t)  with T = 2π/Ω = driving period.

See Also
--------
rtrg.py
"""

from .rtrg import *

OVERWRITE_LEFT = bytes((1,))
OVERWRITE_RIGHT = bytes((2,))
OVERWRITE_BOTH = bytes((3,))


class SymRGfunction(RGfunction):
    """
    Subclass of :class:`RGfunction` for Floquet matrices representing
    functions which in time-domain fulfill :math:`f(t + T/2) = \pm f(t)`.
    """
    def __init__(self, global_properties, values, symmetry=0, diag=None, offdiag=None, **kwargs):
        self.global_properties = global_properties
        self.symmetry = symmetry
        self.energy_shifted_copies = {}
        self.voltage_shifts = 0
        for (key, value) in kwargs.items():
            setattr(self, key, value)
        assert global_properties.voltage_branches == 0

        self.submatrix00 = None
        self.submatrix01 = None
        self.submatrix10 = None
        self.submatrix11 = None
        for (key, value) in kwargs.items():
            setattr(self, key, value)
        if (type(values) == str and values == 'identity'):
            self.symmetry = 1
            self.submatrix00 = np.identity(self.nmax+1, dtype=np.complex128)
            self.submatrix11 = np.identity(self.nmax, dtype=np.complex128)
        elif values is not None:
            if diag is None or offdiag is None:
                self.values = values
            else:
                self.setValues(values, diag, offdiag)

    @classmethod
    def fromRGfunction(cls, rgfunc, diag=None, offdiag=None):
        assert isinstance(rgfunc, RGfunction)
        return SymRGfunction(
                rgfunc.global_properties,
                rgfunc.values,
                rgfunc.symmetry,
                diag,
                offdiag,
                )

    @property
    def values(self):
        values = np.zeros((2*self.nmax+1, 2*self.nmax+1), dtype=np.complex128)
        if self.submatrix00 is not None:
            values[0::2,0::2] = self.submatrix00
        if self.submatrix01 is not None:
            values[0::2,1::2] = self.submatrix01
        if self.submatrix10 is not None:
            values[1::2,0::2] = self.submatrix10
        if self.submatrix11 is not None:
            values[1::2,1::2] = self.submatrix11
        return values

    def setValues(self, values, diag:bool, offdiag:bool):
        """
        More efficient than setting values by self.values = values if
        it is known which parts are non-zero.
        """
        if diag:
            self.submatrix00 = values[0::2,0::2]  # (n+1)x(n+1) matrix for +
            self.submatrix11 = values[1::2,1::2]  #    n x n    matrix for +
        if offdiag:
            self.submatrix01 = values[0::2,1::2]  # (n+1)x n    matrix for -
            self.submatrix10 = values[1::2,0::2]  #    n x(n+1) matrix for -

    @values.setter
    def values(self, values):
        values = np.asarray(values, np.complex128)
        assert values.ndim == 2
        assert values.shape[0] == values.shape[1] == 2*self.nmax+1
        self.submatrix00 = values[0::2,0::2]  # (n+1)x(n+1) matrix for +
        self.submatrix01 = values[0::2,1::2]  # (n+1)x n    matrix for -
        self.submatrix10 = values[1::2,0::2]  #    n x(n+1) matrix for -
        self.submatrix11 = values[1::2,1::2]  #    n x n    matrix for +
        if np.all(np.abs(self.submatrix00) < 1e-15):
            self.submatrix00 = None
        if np.all(np.abs(self.submatrix01) < 1e-15):
            self.submatrix01 = None
        if np.all(np.abs(self.submatrix10) < 1e-15):
            self.submatrix10 = None
        if np.all(np.abs(self.submatrix11) < 1e-15):
            self.submatrix11 = None
        assert (self.submatrix00 is None and self.submatrix11 is None) or (self.submatrix01 is None and self.submatrix10 is None)
        self.energy_shifted_copies.clear()

    def copy(self):
        """
        Copy only values, take a reference to global_properties.
        """
        return SymRGfunction(
                self.global_properties,
                values = None,
                symmetry = self.symmetry,
                submatrix00 = None if self.submatrix00 is None else self.submatrix00.copy(),
                submatrix01 = None if self.submatrix01 is None else self.submatrix01.copy(),
                submatrix10 = None if self.submatrix10 is None else self.submatrix10.copy(),
                submatrix11 = None if self.submatrix11 is None else self.submatrix11.copy(),
            )

    def reduced(self, shift=0):
        """
        Remove voltage-shifted copies
        """
        assert shift == 0
        return self

    def reduced_to_voltage_branches(self, voltage_branches):
        return self

    def floquetConjugate(self):
        """
        For a Floquet matrix A(E)_{nm} this returns the C-transform
            C A(E)_{nm} C = A(-E*)_{-n,-m}
        with the superoperator C defined by
            C x := x^\dag.
        This uses the symmetry of self if self has a symmetry. If this
        C-transform leaves self invariant, this function will return a
        copy of self, but never a reference to self.

        This can only be evaluated if the energy of self lies on the
        imaginary axis.
        """
        if self.symmetry == 1:
            return self.copy()
        elif self.symmetry == -1:
            return -self
        assert abs(self.energy.real) < 1e-12
        return SymRGfunction(
                self.global_properties,
                values = None,
                submatrix00 = None if self.submatrix00 is None else np.conjugate(self.submatrix00[::-1,::-1]),
                submatrix01 = None if self.submatrix01 is None else np.conjugate(self.submatrix01[::-1,::-1]),
                submatrix10 = None if self.submatrix10 is None else np.conjugate(self.submatrix10[::-1,::-1]),
                submatrix11 = None if self.submatrix11 is None else np.conjugate(self.submatrix11[::-1,::-1]),
            )

    def __matmul__(self, other):
        """
        Convolution (or product in Floquet space) of two RG functions.
        Other must be of type SymRGfunction.

        Note: This is only approximately associative, as long the function
        converges to 0 for |n| of order of nmax.
        """
        if not isinstance(other, SymRGfunction):
            if isinstance(other, RGfunction):
                return self.toRGfunction() @ other
            return NotImplemented
        assert self.global_properties is other.global_properties
        res00 = None
        res11 = None
        res01 = None
        res10 = None
        symmetry = self.symmetry * other.symmetry;
        if (self.submatrix00 is not None and other.submatrix00 is not None):
            assert (self.submatrix11 is not None and other.submatrix11 is not None)
            res00 = rtrg_c.multiply_extended(other.submatrix00.T, self.submatrix00.T, self.padding//2, symmetry, self.clear_corners//2).T
            res11 = rtrg_c.multiply_extended(other.submatrix11.T, self.submatrix11.T, self.padding//2, symmetry, self.clear_corners//2).T
        elif (self.submatrix01 is not None and other.submatrix01 is not None):
            assert (self.submatrix10 is not None and other.submatrix10 is not None)
            res00 = rtrg_c.multiply_extended(other.submatrix10.T, self.submatrix01.T, self.padding//2, symmetry, self.clear_corners//2).T
            res11 = rtrg_c.multiply_extended(other.submatrix01.T, self.submatrix10.T, self.padding//2, symmetry, self.clear_corners//2).T
        elif (self.submatrix00 is not None and other.submatrix01 is not None):
            assert (self.submatrix11 is not None and other.submatrix10 is not None)
            res01 = rtrg_c.multiply_extended(other.submatrix01.T, self.submatrix00.T, self.padding//2, symmetry, self.clear_corners//2).T
            res10 = rtrg_c.multiply_extended(other.submatrix10.T, self.submatrix11.T, self.padding//2, symmetry, self.clear_corners//2).T
        elif (self.submatrix01 is not None and other.submatrix00 is not None):
            assert (self.submatrix10 is not None and other.submatrix11 is not None)
            res01 = rtrg_c.multiply_extended(other.submatrix11.T, self.submatrix01.T, self.padding//2, symmetry, self.clear_corners//2).T
            res10 = rtrg_c.multiply_extended(other.submatrix00.T, self.submatrix10.T, self.padding//2, symmetry, self.clear_corners//2).T
        return SymRGfunction(
                self.global_properties,
                values = None,
                submatrix00 = res00,
                submatrix01 = res01,
                submatrix10 = res10,
                submatrix11 = res11,
                symmetry = self.symmetry * other.symmetry,
            )

    def __rmatmul__(self, other):
        if isinstance(other, SymRGfunction):
            return other @ self
        if isinstance(other, RGfunction):
            return other @ self.toRGfunction()
        return NotImplemented

    def __imatmul__(self, other):
        if not isinstance(other, RGfunction):
            return NotImplemented
        assert self.global_properties is other.global_properties
        self.symmetry *= other.symmetry
        symmetry = self.symmetry;
        if (self.submatrix00 is not None and other.submatrix00 is not None):
            assert (self.submatrix11 is not None and other.submatrix11 is not None)
            self.submatrix00 = rtrg_c.multiply_extended(other.submatrix00.T, self.submatrix00.T, self.padding//2, symmetry, self.clear_corners//2, OVERWRITE_LEFT).T
            self.submatrix11 = rtrg_c.multiply_extended(other.submatrix11.T, self.submatrix11.T, self.padding//2, symmetry, self.clear_corners//2, OVERWRITE_LEFT).T
        elif (self.submatrix01 is not None and other.submatrix01 is not None):
            assert (self.submatrix10 is not None and other.submatrix10 is not None)
            self.submatrix00 = rtrg_c.multiply_extended(other.submatrix10.T, self.submatrix01.T, self.padding//2, symmetry, self.clear_corners//2, OVERWRITE_LEFT).T
            self.submatrix11 = rtrg_c.multiply_extended(other.submatrix01.T, self.submatrix10.T, self.padding//2, symmetry, self.clear_corners//2, OVERWRITE_LEFT).T
            self.submatrix01 = None
            self.submatrix10 = None
        elif (self.submatrix00 is not None and other.submatrix01 is not None):
            assert (self.submatrix11 is not None and other.submatrix10 is not None)
            self.submatrix01 = rtrg_c.multiply_extended(other.submatrix01.T, self.submatrix00.T, self.padding//2, symmetry, self.clear_corners//2, OVERWRITE_LEFT).T
            self.submatrix10 = rtrg_c.multiply_extended(other.submatrix10.T, self.submatrix11.T, self.padding//2, symmetry, self.clear_corners//2, OVERWRITE_LEFT).T
            self.submatrix00 = None
            self.submatrix11 = None
        elif (self.submatrix01 is not None and other.submatrix00 is not None):
            assert (self.submatrix10 is not None and other.submatrix11 is not None)
            self.submatrix01 = rtrg_c.multiply_extended(other.submatrix11.T, self.submatrix01.T, self.padding//2, symmetry, self.clear_corners//2, OVERWRITE_LEFT).T
            self.submatrix10 = rtrg_c.multiply_extended(other.submatrix00.T, self.submatrix10.T, self.padding//2, symmetry, self.clear_corners//2, OVERWRITE_LEFT).T
        return self

    def __add__(self, other):
        """
        Add other to self. If other is a scalar or a scalar function of energy
        represented by an array of values at self.energies, this treats other
        as an identity (or diagonal) Floquet matrix.
        Other must be a scalar or array of same shape as self.energies or an
        RGfunction of the same shape and energies as self.
        """
        if isinstance(other, SymRGfunction):
            assert self.global_properties is other.global_properties
            symmetry = (self.symmetry == other.symmetry) * self.symmetry
            assert (self.submatrix00 is None and other.submatrix00 is None) or (self.submatrix01 is None and other.submatrix01 is None)
            return SymRGfunction(
                    self.global_properties,
                    values = None,
                    submatrix00 = other.submatrix00 if self.submatrix00 is None else (self.submatrix00 if other.submatrix00 is None else self.submatrix00 + other.submatrix00),
                    submatrix01 = other.submatrix01 if self.submatrix01 is None else (self.submatrix01 if other.submatrix01 is None else self.submatrix01 + other.submatrix01),
                    submatrix10 = other.submatrix10 if self.submatrix10 is None else (self.submatrix10 if other.submatrix10 is None else self.submatrix10 + other.submatrix10),
                    submatrix11 = other.submatrix11 if self.submatrix11 is None else (self.submatrix11 if other.submatrix11 is None else self.submatrix11 + other.submatrix11),
                    symmetry = symmetry,
                )
        elif isinstance(other, RGfunction):
            return self.toRGfunction() + other
        elif np.shape(other) == () or np.shape(other) == (2*self.nmax+1,):
            assert self.submatrix01 is None and self.submatrix10 is None
            # TODO: symmetries
            # Assume that other represents a (possibly energy-dependent) scalar.
            res00 = self.submatrix00.copy()
            res11 = self.submatrix11.copy()
            symmetry = 0
            if isinstance(other, Number):
                if self.symmetry == 1 and other.imag == 0:
                    symmetry = 1
                elif self.symmetry == -1 and other.real == 0:
                    symmetry = -1
            try:
                res00[np.diag_indices(self.nmax+1)] += other
                res11[np.diag_indices(self.nmax+1)] += other
            except:
                res00[np.diag_indices(self.nmax+1)] += other[0::2]
                res11[np.diag_indices(self.nmax+1)] += other[1::2]
            return SymRGfunction(
                    self.global_properties,
                    values = None,
                    submatrix00 = res00,
                    submatrix01 = None,
                    submatrix10 = None,
                    submatrix11 = res11,
                    symmetry = symmetry,
                )
        else:
            raise TypeError("unsupported operand types for +: RGfunction and", type(other))

    def __sub__(self, other):
        if isinstance(other, SymRGfunction):
            assert self.global_properties is other.global_properties
            symmetry = (self.symmetry == other.symmetry) * self.symmetry
            assert (self.submatrix00 is None and other.submatrix00 is None) or (self.submatrix01 is None and other.submatrix01 is None)
            return SymRGfunction(
                    self.global_properties,
                    values = None,
                    submatrix00 = (None if other.submatrix00 is None else -other.submatrix00) if self.submatrix00 is None else (self.submatrix00 if other.submatrix00 is None else self.submatrix00 - other.submatrix00),
                    submatrix01 = (None if other.submatrix01 is None else -other.submatrix01) if self.submatrix01 is None else (self.submatrix01 if other.submatrix01 is None else self.submatrix01 - other.submatrix01),
                    submatrix10 = (None if other.submatrix10 is None else -other.submatrix10) if self.submatrix10 is None else (self.submatrix10 if other.submatrix10 is None else self.submatrix10 - other.submatrix10),
                    submatrix11 = (None if other.submatrix11 is None else -other.submatrix11) if self.submatrix11 is None else (self.submatrix11 if other.submatrix11 is None else self.submatrix11 - other.submatrix11),
                    symmetry = symmetry,
                )
        if isinstance(other, RGfunction):
            return self.toRGfunction() - other
        elif np.shape(other) == () or np.shape(other) == (2*self.nmax+1,):
            assert self.submatrix01 is None and self.submatrix10 is None
            # TODO: symmetries
            # Assume that other represents a (possibly energy-dependent) scalar.
            res00 = self.submatrix00.copy()
            res11 = self.submatrix11.copy()
            symmetry = 0
            if isinstance(other, Number):
                if self.symmetry == 1 and other.imag == 0:
                    symmetry = 1
                elif self.symmetry == -1 and other.real == 0:
                    symmetry = -1
            try:
                res00[np.diag_indices(self.nmax+1)] -= other
                res11[np.diag_indices(self.nmax+1)] -= other
            except:
                res00[np.diag_indices(self.nmax+1)] -= other[0::2]
                res11[np.diag_indices(self.nmax+1)] -= other[1::2]
            return SymRGfunction(
                    self.global_properties,
                    values = None,
                    submatrix00 = res00,
                    submatrix01 = None,
                    submatrix10 = None,
                    submatrix11 = res11,
                    symmetry = symmetry,
                )
        else:
            raise TypeError("unsupported operand types for +: RGfunction and", type(other))

    def __neg__(self):
        """
        Return a copy of self with inverted sign of self.values.
        """
        return SymRGfunction(
                self.global_properties,
                values = None,
                submatrix00 = None if self.submatrix00 is None else -self.submatrix00,
                submatrix01 = None if self.submatrix01 is None else -self.submatrix01,
                submatrix10 = None if self.submatrix10 is None else -self.submatrix10,
                submatrix11 = None if self.submatrix11 is None else -self.submatrix11,
                symmetry = self.symmetry,
            )

    def __iadd__(self, other):
        if isinstance(other, SymRGfunction):
            assert self.global_properties is other.global_properties
            self.symmetry = (self.symmetry == other.symmetry) * self.symmetry
            assert (self.submatrix00 is None and other.submatrix00 is None) or (self.submatrix01 is None and other.submatrix01 is None)
            if (self.submatrix01 is None):
                self.submatrix00 += other.submatrix00
                self.submatrix11 += other.submatrix11
            else:
                self.submatrix01 += other.submatrix01
                self.submatrix10 += other.submatrix10
        elif np.shape(other) == () or np.shape(other) == (2*self.nmax+1,):
            assert self.submatrix01 is None and self.submatrix10 is None
            # TODO: symmetries
            # Assume that other represents a (possibly energy-dependent) scalar.
            if self.submatrix00 is None:
                self.submatrix00 = np.zeros((self.nmax+1, self.nmax+1), np.complex128)
            if self.submatrix11 is None:
                self.submatrix11 = np.zeros((self.nmax, self.nmax), np.complex128)
            try:
                self.submatrix00[np.diag_indices(self.nmax+1)] += other
                self.submatrix11[np.diag_indices(self.nmax)] += other
            except:
                self.submatrix00[np.diag_indices(self.nmax+1)] += other[0::2]
                self.submatrix11[np.diag_indices(self.nmax)] += other[1::2]
            if not isinstance(other, Number) or not ((self.symmetry == 1 and other.imag == 0) or (self.symmetry == -1 and other.real == 0)):
                self.symmetry = 0
        else:
            raise TypeError("unsupported operand types for +: RGfunction and", type(other))
        return self

    def __isub__(self, other):
        if isinstance(other, SymRGfunction):
            assert self.global_properties is other.global_properties
            self.symmetry = (self.symmetry == other.symmetry) * self.symmetry
            assert (self.submatrix00 is None and other.submatrix00 is None) or (self.submatrix01 is None and other.submatrix01 is None)
            if (self.submatrix01 is None):
                self.submatrix00 -= other.submatrix00
                self.submatrix11 -= other.submatrix11
            else:
                self.submatrix01 -= other.submatrix01
                self.submatrix10 -= other.submatrix10
        elif np.shape(other) == () or np.shape(other) == (2*self.nmax+1,):
            assert self.submatrix01 is None and self.submatrix10 is None
            # TODO: symmetries
            # Assume that other represents a (possibly energy-dependent) scalar.
            try:
                self.submatrix00[np.diag_indices(self.nmax+1)] -= other
                self.submatrix11[np.diag_indices(self.nmax+1)] -= other
            except:
                self.submatrix00[np.diag_indices(self.nmax+1)] -= other[0::2]
                self.submatrix11[np.diag_indices(self.nmax+1)] -= other[1::2]
            self.symmetry = 0
        else:
            raise TypeError("unsupported operand types for +: RGfunction and", type(other))
        return self

    def __mul__(self, other):
        """
        Multiplication by scalar or SymRGfunction.
        If other is a SymRGfunction, this calculates the matrix product
        (equivalent to __matmul__). If other is a scalar, this multiplies
        self.values by other.
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
            return SymRGfunction(
                    self.global_properties,
                    values = None,
                    submatrix00 = None if self.submatrix00 is None else other*self.submatrix00,
                    submatrix01 = None if self.submatrix01 is None else other*self.submatrix01,
                    submatrix10 = None if self.submatrix10 is None else other*self.submatrix10,
                    submatrix11 = None if self.submatrix11 is None else other*self.submatrix11,
                    symmetry = symmetry,
                )
        return NotImplemented

    def __imul__(self, other):
        """
        In-place multiplication by scalar or SymRGfunction.
        If other is a SymRGfunction, this calculates the matrix product
        (equivalent to __matmul__). If other is a scalar, this multiplies
        self.values by other.
        """
        if isinstance(other, SymRGfunction):
            self @= other
        elif isinstance(other, Number):
            if other.imag != 0:
                if other.real == 0:
                    self.symmetry = -self.symmetry
                else:
                    self.symmetry = 0
            if (self.submatrix00 is not None):
                self.submatrix00 *= other
            if (self.submatrix01 is not None):
                self.submatrix01 *= other
            if (self.submatrix10 is not None):
                self.submatrix10 *= other
            if (self.submatrix11 is not None):
                self.submatrix11 *= other
        else:
            return NotImplemented
        return self

    def __rmul__(self, other):
        """
        Reverse multiplication by scalar or RGfunction.
        If other is an RGfunction, this calculates the matrix product
        (equivalent to __matmul__). If other is a scalar, this multiplies
        self.values by other.
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
            return self * other
        return NotImplemented

    def __truediv__(self, other):
        """
        Divide self by other, which must be a scalar.
        """
        if isinstance(other, Number):
            return self * (1/other)
        return NotImplemented

    def __itruediv__(self, other):
        """
        Divide self in-place by other, which must be a scalar.
        """
        if isinstance(other, Number):
            self *= (1/other)
            return self
        return NotImplemented

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __repr__(self):
        return 'SymRGfunction{ %s,\n00: %s\n01: %s\n10: %s\n11: %s }'%(self.energy, self.submatrix00.__repr__(), self.submatrix01.__repr__(), self.submatrix10.__repr__(), self.submatrix11.__repr__())

    def __getitem__(self, arg):
        raise NotImplementedError

    def __eq__(self, other):
        return ( self.global_properties is other.global_properties ) \
                and (self.submatrix00 is other.submatrix00 or np.allclose(self.submatrix00, other.submatrix00)) \
                and (self.submatrix01 is other.submatrix01 or np.allclose(self.submatrix01, other.submatrix01)) \
                and (self.submatrix10 is other.submatrix10 or np.allclose(self.submatrix10, other.submatrix10)) \
                and (self.submatrix11 is other.submatrix11 or np.allclose(self.submatrix11, other.submatrix11)) \
                and self.symmetry == other.symmetry

    def k2lambda(self, shift_matrix=None):
        """
        Assume that self is K_n^m(E) = K_n(E-mΩ).
        Then calculate Λ_n^m(E) such that (approximately)

                     m    [                   m-k    ]
            δ   = Σ Λ (E) [ (E-(m-n)Ω) δ   - K   (E) ] .
             n0   k  k    [             kn    n-k    ]

        This calculates the propagator from an effective Liouvillian.
        Some of the linear systems of equation which we need to solve here are
        overdetermined. This means that we can in general only get an
        approximate solution because an exact solution does not exist.

        TODO: implement direct energy shift?
        """
        assert shift_matrix is None
        assert self.submatrix01 is None and self.submatrix10 is None
        invert = -self
        invert.submatrix00[np.diag_indices(self.nmax+1)] += self.energy + self.omega*np.arange(-self.nmax, self.nmax+1, 2)
        invert.submatrix11[np.diag_indices(self.nmax)] += self.energy + self.omega*np.arange(-self.nmax+1, self.nmax+1, 2)
        invert.symmetry = -1 if self.symmetry == -1 else 0
        return invert.inverse()

    def inverse(self):
        """
        For a given object self = A try to calculate B such that
        A @ B = identity
        with identity[n,k] = δ(n,0).

        Some of the linear systems of equation which we need to solve here are
        overdetermined. This means that we can in general only get an
        approximate solution because an exact solution does not exist.
        """

        assert self.submatrix01 is None and self.submatrix10 is None
        assert self.submatrix00 is not None and self.submatrix11 is not None
        try:
            res00 = rtrg_c.invert_extended(self.submatrix00.T, self.padding//2, round(settings.LAZY_INVERSE_FACTOR*self.padding/2)).T
            res11 = rtrg_c.invert_extended(self.submatrix11.T, self.padding//2, round(settings.LAZY_INVERSE_FACTOR*self.padding/2)).T
        except:
            settings.logger.exception("padded inversion failed in compact RTRG")
            res00 = np.linalg.inv(self.submatrix00)
            res11 = np.linalg.inv(self.submatrix11)
        return SymRGfunction(
                self.global_properties,
                values = None,
                submatrix00 = res00,
                submatrix01 = None,
                submatrix10 = None,
                submatrix11 = res11,
                symmetry = self.symmetry,
            )

    def toRGfunction(self):
        return RGfunction(self.global_properties, self.values, symmetry=self.symmetry)

    def shift_energies(self, n=0):
        raise ValueError("shift_energies is not defined for SymRGfunction")

    def check_symmetry(self):
        assert self.symmetry in (-1,0,1)
        if self.symmetry:
            if self.submatrix11 is not None:
                assert np.allclose(self.submatrix11[::-1,::-1].conjugate(), self.symmetry*self.submatrix11)
            if self.submatrix00 is not None:
                assert np.allclose(self.submatrix00[::-1,::-1].conjugate(), self.symmetry*self.submatrix00)
            if self.submatrix01 is not None:
                assert np.allclose(self.submatrix01[::-1,::-1].conjugate(), self.symmetry*self.submatrix01)
            if self.submatrix10 is not None:
                assert np.allclose(self.submatrix10[::-1,::-1].conjugate(), self.symmetry*self.submatrix10)
