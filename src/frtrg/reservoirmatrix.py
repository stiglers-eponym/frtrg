# Copyright 2021 Valentin Bruch <valentin.bruch@rwth-aachen.de>
# License: MIT
"""
Kondo FRTRG, module defining vertice in RG equations

Module defining class ReservoirMatrix and some functions for efficient
handling of matrices of Floquet matrices as used for the Kondo model.

See also: rtrg.py
"""

import numpy as np
from numbers import Number
from . import settings
from .rtrg import RGobj, RGfunction


class ReservoirMatrix(RGobj):
    """
    2x2 matrix of RGfunctions.
    This includes a system of book keeping for energy shifts by
    multiples of the voltage in products of ReservoirMatrices.
    if symmetry != 0:
        self.data[1,0] == symmetry * self.data[0,1].floquetConjugate()
    if symmety != 0 and global_properties.symmetric:
        self.data[0,0] == self.data[1,1]

    Multiplication operators for ReservoirMatrix objects are:
    *   for multiplication with scalars
    @   for multiplication with RGfunctions and normal matrix
        multiplication with ReservoirMatrices
        (matrix indices ij, jk → ik with sum over j)
    %   for transpose matrix multiplication with ReservoirMatrices
        (matrix indices jk, ij → ik with sum over j)
                      ⎛ ⊤   ⊤⎞⊤
        i.e.  A % B = ⎜A @ B ⎟  if there are no energy argument shifts.
                      ⎝      ⎠
    """

    def __init__(self, global_properties, symmetry=0):
        super().__init__(global_properties, symmetry)
        self.data = np.ndarray((2, 2), dtype=RGfunction)
        self.voltage_shifts = 0

    def __getitem__(self, arg):
        assert self.data[arg].voltage_shifts == self.voltage_shifts + arg[1] - arg[0]
        return self.data[arg]

    def __setitem__(self, indices, value):
        self.data.__setitem__(indices, value)
        if isinstance(value, RGfunction):
            assert value.global_properties is self.global_properties
            self.data.__getitem__(indices).voltage_shifts = \
                    self.voltage_shifts + indices[1] - indices[0]

    def __add__(self, other):
        if isinstance(other, ReservoirMatrix):
            assert self.global_properties is other.global_properties
            assert self.voltage_shifts == other.voltage_shifts
            symmetry = (self.symmetry == other.symmetry) * self.symmetry
            res = ReservoirMatrix(self.global_properties, symmetry)
            res.voltage_shifts = self.voltage_shifts
            res.data = self.data + other.data
            return res
        else:
            raise NotImplementedError(
                    'Addition is not defined for types %s and %s'%(
                        ReservoirMatrix, type(other)))

    def __iadd__(self, other):
        if isinstance(other, ReservoirMatrix):
            assert self.global_properties is other.global_properties
            assert self.voltage_shifts == other.voltage_shifts
            self.data += other.data
            if self.symmetry != other.symmetry:
                self.symmetry = 0
        else:
            raise NotImplementedError(
                    'Addition is not defined for types %s and %s'%(
                        ReservoirMatrix, type(other)))
        return self

    def __sub__(self, other):
        if isinstance(other, ReservoirMatrix):
            assert self.global_properties is other.global_properties
            assert self.voltage_shifts == other.voltage_shifts
            symmetry = (self.symmetry == other.symmetry) * self.symmetry
            res = ReservoirMatrix(self.global_properties, symmetry)
            res.voltage_shifts = self.voltage_shifts
            res.data = self.data - other.data
            return res
        else:
            raise NotImplementedError(
                    'Subtraction is not defined for types %s and %s'%(
                        ReservoirMatrix, type(other)))

    def __isub__(self, other):
        if isinstance(other, ReservoirMatrix):
            assert self.global_properties is other.global_properties
            assert self.voltage_shifts == other.voltage_shifts
            self.data -= other.data
            if self.symmetry != other.symmetry:
                self.symmetry = 0
        else:
            raise NotImplementedError(
                    'Subtraction is not defined for types %s and %s'%(
                        ReservoirMatrix, type(other)))
        return self

    def __neg__(self):
        res = ReservoirMatrix(self.global_properties, self.symmetry)
        res.voltage_shifts = voltage_shifts
        res.data = -self.data
        return res

    def __mul__(self, other):
        if isinstance(other, ReservoirMatrix) or isinstance(other, RGfunction):
            raise TypeError('Multiplication of reservoir matrices uses the @ symbol.')
        else:
            res = ReservoirMatrix(self.global_properties)
            if other.imag == 0:
                res.symmetry = self.symmetry
            elif other.real == 0:
                res.symmetry = -self.symmetry
            else:
                res.symmetry = 0
            res.data = self.data * other
        return res

    def __imul__(self, other):
        if isinstance(other, ReservoirMatrix):
            raise TypeError('Multiplication of reservoir matrices uses the @ symbol.')
        else:
            if other.imag != 0:
                if other.real == 0:
                    self.symmetry = -self.symmetry
                else:
                    self.symmetry = 0
            self.data *= other
        return self

    def __rmul__(self, other):
        if isinstance(other, ReservoirMatrix):
            raise TypeError('Multiplication of reservoir matrices uses the @ symbol.')
        elif isinstance(other, RGfunction):
            res = ReservoirMatrix(self.global_properties, other.symmetry * self.symmetry)
            res.data = other * self.data
        elif isinstance(other, Number):
            res = ReservoirMatrix(self.global_properties)
            res.data = other * self.data
            if other.imag == 0:
                res.symmetry = self.symmetry
            elif other.real == 0:
                res.symmetry = -self.symmetry
            else:
                res.symmetry = 0
        return res

    def __matmul__(self, other):
        if isinstance(other, ReservoirMatrix):
            # 8 multiplications without symmetry
            # 7 multiplications with symmetry but xL != xR
            # 4 multiplications with symmetry and xL == xR
            assert other.voltage_shifts == 0
            assert self.global_properties is other.global_properties
            res = ReservoirMatrix(self.global_properties, symmetry=0)
            res.voltage_shifts = self.voltage_shifts

            res_00_00 = self[0,0] @ other[0,0]
            res_01_10 = self[0,1] @ other[1,0]
            res[0,0] = res_00_00 + res_01_10
            res[0,1] = self[0,0] @ other[0,1] + self[0,1] @ other[1,1]
            symmetry = self.symmetry * other.symmetry * (self.voltage_shifts == 0)
            if symmetry == 0 or settings.IGNORE_SYMMETRIES:
                res[1,1] = self[1,0] @ other[0,1] + self[1,1] @ other[1,1]
                res[1,0] = self[1,0] @ other[0,0] + self[1,1] @ other[1,0]
            elif self.global_properties.symmetric:
                res[1,1] = res_00_00 + symmetry * res_01_10.floquetConjugate()
                res[1,0] = symmetry * res[0,1].floquetConjugate()
            else:
                res[1,1] = self[1,1] @ other[1,1] + symmetry * res_01_10.floquetConjugate()
                res[1,0] = self[1,0] @ other[0,0] + self[1,1] @ other[1,0]
            return res
        elif isinstance(other, RGfunction):
            # 4 multiplications without symmetry
            # 3 multiplications with symmetry but xL != xR
            # 2 multiplications with symmetry and xL == xR
            assert self.global_properties is other.global_properties
            res = ReservoirMatrix(
                    self.global_properties,
                    self.symmetry * other.symmetry * (self.voltage_shifts == 0))
            res.voltage_shifts = self.voltage_shifts + other.voltage_shifts
            res[0,0] = self[0,0] @ other
            if res.symmetry == 0 or settings.IGNORE_SYMMETRIES:
                res[0,1] = self[0,1] @ other
                res[1,0] = self[1,0] @ other
                res[1,1] = self[1,1] @ other
            else:
                if res.global_properties.symmetric:
                    res[1,1] = res[0,0].copy()
                else:
                    res[1,1] = self[1,1] @ other
                res[0,1] = self[0,1] @ other
                res[1,0] = res.symmetry * res[0,1].floquetConjugate()
            return res
        else:
            raise TypeError(
                    'Math multiplication is not defined for types %s and %s'%(
                        ReservoirMatrix, type(other)))

    def __rmatmul__(self, other):
        if isinstance(other, RGfunction):
            assert self.voltage_shifts == 0
            assert self.global_properties is other.global_properties
            res = ReservoirMatrix(
                    self.global_properties,
                    self.symmetry * other.symmetry * (other.voltage_shifts == 0))
            res.voltage_shifts = other.voltage_shifts
            res[0,0] = other @ self[0,0]
            if res.symmetry == 0 or settings.IGNORE_SYMMETRIES:
                res[0,1] = other @ self[0,1]
                res[1,0] = other @ self[1,0]
                res[1,1] = other @ self[1,1]
            else:
                if res.global_properties.symmetric:
                    res[1,1] = res[0,0].copy()
                else:
                    res[1,1] = other @ self[1,1]
                res[0,1] = other @ self[0,1]
                res[1,0] = res.symmetry * res[0,1].floquetConjugate()
            return res
        else:
            raise TypeError(
                    'Math multiplication is not defined for types %s and %s'%(
                        type(other), ReservoirMatrix))

    def __imatmul__(self, other):
        if isinstance(other, ReservoirMatrix):
            # 8 multiplications without symmetry
            # 7 multiplications with symmetry but xL != xR
            # 4 multiplications with symmetry and xL == xR
            assert other.voltage_shifts == 0
            assert self.global_properties is other.global_properties
            res_00_00 = self[0,0] @ other[0,0]
            res_01_10 = self[0,1] @ other[1,0]
            res_00 = res_00_00 + res_01_10
            res_01 = self[0,0] @ other[0,1] + self[0,1] @ other[1,1]
            symmetry = self.symmetry * other.symmetry
            if symmetry == 0 or settings.IGNORE_SYMMETRIES:
                res_11 = self[1,0] @ other[0,1] + self[1,1] @ other[1,1]
                res_10 = self[1,0] @ other[0,0] + self[1,1] @ other[1,0]
            elif self.global_properties.symmetric:
                res_11 = res_00_00 + symmetry * res_01_10.floquetConjugate()
                res_10 = symmetry * res_01.floquetConjugate()
            else:
                res_11 = self[1,1] @ other[1,1] + symmetry * res_01_10.floquetConjugate()
                res_10 = self[1,0] @ other[0,0] + self[1,1] @ other[1,0]
            self[0,0] = res_00
            self[0,1] = res_01
            self[1,0] = res_10
            self[1,1] = res_11
            self.symmetry = 0
            return self
        elif isinstance(other, RGfunction):
            # TODO: this does not take into account any symmetries
            for idx in ((0,0), (0,1), (1,0), (1,1)):
                self[idx] @= other
            self.symmetry = 0
            return self
        else:
            raise TypeError(
                    'Math multiplication is not defined for types %s and %s'%(
                        ReservoirMatrix, type(other)))

    def __mod__(self, other):
        """
        Transpose multiplication: Given A, B return C such that

            C   = A   B
             12    32  13
        """
        assert isinstance(other, ReservoirMatrix)
        assert other.voltage_shifts == 0
        res = ReservoirMatrix(self.global_properties, symmetry=0)
        res.voltage_shifts = self.voltage_shifts

        res_00_00 = self[0,0] @ other[0,0]
        res_10_01 = self[1,0] @ other[0,1]
        res[0,0] = res_00_00 + res_10_01
        res[0,1] = self[0,1] @ other[0,0] + self[1,1] @ other[0,1]
        # TODO: check symmetry
        symmetry = self.symmetry * other.symmetry * (self.voltage_shifts == 0)
        if symmetry == 0 or settings.IGNORE_SYMMETRIES:
            res[1,1] = self[0,1] @ other[1,0] + self[1,1] @ other[1,1]
            res[1,0] = self[0,0] @ other[1,0] + self[1,0] @ other[1,1]
        elif self.global_properties.symmetric:
            # TODO: check symmetric case
            res[1,1] = res_00_00 + symmetry * res_10_01.floquetConjugate()
            res[1,0] = symmetry * res[0,1].floquetConjugate()
        else:
            res[1,1] = self[1,1] @ other[1,1] + symmetry * res_10_01.floquetConjugate()
            res[1,0] = self[0,0] @ other[1,0] + self[1,0] @ other[1,1]
        return res

    def tr(self):
        return self[0,0] + self[1,1]

    def copy(self):
        res = ReservoirMatrix(self.global_properties, self.symmetry)
        res.voltage_shifts = self.voltage_shifts
        for i in range(2):
            for j in range(2):
                res[i,j] = self[i,j].copy()
        return res

    def __eq__(self, other):
        return self.global_properties is other.global_properties \
                and np.all(self.data == other.data)

    def to_numpy_array(self):
        array = np.ndarray((2,2,*self[0,0].values.shape), dtype=np.complex128)
        for i in range(2):
            for j in range(2):
                array[i,j] = self[i,j].values
        return array

    def check_symmetry(self):
        assert self.symmetry in (-1,0,1)
        if self.global_properties.symmetric:
            assert np.allclose(self[0,0].values, self[1,1].values)
        if self.symmetry:
            assert np.allclose(self[0,1].values, self.symmetry*self[1,0].floquetConjugate().values)
            self[0,0].check_symmetry()
            self[1,1].check_symmetry()



def einsum_34_12_43(a:ReservoirMatrix, b:RGobj, c:ReservoirMatrix) -> RGobj:
    """
    Compute
        A_34 B_12 C_43
    with (implicit) summation over indices 3 and 4.
    B can be either a scalar or a reservoir matrix.

    8 multiplications if b is a scalar,
    32 multiplications if b is a reservoir matrix without symmetry,
    20 multiplications if b is a reservoir matrix with symmetry and xL != xR,
    10 multiplications if b is a reservoir matrix with symmetry and xL == xR.
    """
    assert isinstance(a, ReservoirMatrix)
    assert isinstance(b, RGobj)
    assert isinstance(c, ReservoirMatrix)
    assert a.global_properties is b.global_properties is c.global_properties
    assert c.voltage_shifts == 0
    symmetry = a.symmetry * b.symmetry * c.symmetry
    if symmetry == 0 or settings.IGNORE_SYMMETRIES:
        if settings.ENFORCE_SYMMETRIC:
            raise RuntimeError(
                    'Unsymmetric einsum_34_12_43: %d %d %d'%(
                        a.symmetry, b.symmetry, c.symmetry))
        return a[0,0] @ b @ c[0,0] \
                + a[0,1] @ b @ c[1,0] \
                + a[1,0] @ b @ c[0,1] \
                + a[1,1] @ b @ c[1,1]
    if not isinstance(b, ReservoirMatrix):
        res_01_10 = a[0,1] @ b @ c[1,0]
        if a.global_properties.symmetric:
            res = 2*a[0,0] @ b @ c[0,0] \
                    + res_01_10 \
                    + symmetry * res_01_10.floquetConjugate()
        else:
            res = a[0,0] @ b @ c[0,0] \
                    + a[1,1] @ b @ c[1,1] \
                    + res_01_10 \
                    + symmetry * res_01_10.floquetConjugate()
        res.symmetry = symmetry
        return res
    if a.global_properties.symmetric:
        # xL = xR = 0.5
        res_00_01 = a[0,1] @ b[0,0] @ c[1,0]
        res_00 = 2 * a[0,0] @ b[0,0] @ c[0,0] \
                + res_00_01 \
                + symmetry * res_00_01.floquetConjugate()
        res_01 = 2 * a[0,0] @ b[0,1] @ c[0,0] \
                + a[0,1] @ b[0,1] @ c[1,0] \
                + a[1,0] @ b[0,1] @ c[0,1]
        res = ReservoirMatrix(a.global_properties, symmetry)
        res.voltage_shifts = a.voltage_shifts + b.voltage_shifts + c.voltage_shifts
        res[0,0] = res_00
        res[1,1] = res_00.copy()
        res[0,1] = res_01
        res[1,0] = symmetry * res_01.floquetConjugate()
        return res
    else:
        # TODO: check!
        # xL != xR
        res_00_01 = a[0,1] @ b[0,0] @ c[1,0]
        res_11_01 = a[0,1] @ b[1,1] @ c[1,0]
        res_00 = a[0,0] @ b[0,0] @ c[0,0] \
                + a[1,1] @ b[0,0] @ c[1,1] \
                + res_00_01 \
                + symmetry * res_00_01.floquetConjugate()
        res_11 = a[0,0] @ b[1,1] @ c[0,0] \
                + a[1,1] @ b[1,1] @ c[1,1] \
                + res_11_01 \
                + symmetry * res_11_01.floquetConjugate()
        res_01 = a[1,1] @ b[0,1] @ c[1,1] \
                + a[0,0] @ b[0,1] @ c[0,0] \
                + a[0,1] @ b[0,1] @ c[1,0] \
                + a[1,0] @ b[0,1] @ c[0,1]
        res = ReservoirMatrix(a.global_properties, symmetry)
        res.voltage_shifts = a.voltage_shifts + b.voltage_shifts + c.voltage_shifts
        res[0,0] = res_00
        res[1,1] = res_11
        res[0,1] = res_01
        res[1,0] = symmetry * res_01.floquetConjugate()
        return res


def einsum_34_12_43_double(
        a: ReservoirMatrix,
        b: ReservoirMatrix,
        c: ReservoirMatrix,
        d: ReservoirMatrix
        ) -> (ReservoirMatrix, ReservoirMatrix):
    """
    A_34 B_12 C_43 , A_34 B_12 D_43

    48 multiplications if b is a reservoir matrix,
    30 multiplications with symmetries if xL != xR,
    15 multiplications with symmetries if xL == xR.
    """
    assert isinstance(a, ReservoirMatrix)
    assert isinstance(b, ReservoirMatrix)
    assert isinstance(c, ReservoirMatrix)
    assert isinstance(d, ReservoirMatrix)
    assert a.global_properties \
            is b.global_properties \
            is c.global_properties \
            is d.global_properties
    assert c.voltage_shifts == d.voltage_shifts == 0
    symmetry_c = a.symmetry * b.symmetry * c.symmetry
    symmetry_d = a.symmetry * b.symmetry * d.symmetry
    if symmetry_c == 0 or symmetry_d == 0 or settings.IGNORE_SYMMETRIES:
        if settings.ENFORCE_SYMMETRIC:
            raise RuntimeError(
                'Unsymmetric einsum_34_12_43_double: %d %d %d %d'%(
                    a.symmetry, b.symmetry, c.symmetry, d.symmetry))
        ab_00 = a[0,0] @ b
        ab_01 = a[0,1] @ b
        ab_10 = a[1,0] @ b
        ab_11 = a[1,1] @ b
        return (
                ab_00 @ c[0,0] + ab_01 @ c[1,0] + ab_10 @ c[0,1] + ab_11 @ c[1,1],
                ab_00 @ d[0,0] + ab_01 @ d[1,0] + ab_10 @ d[0,1] + ab_11 @ d[1,1]
                )
    if a.global_properties.symmetric:
        # xL == xR == 0.5
        ab_00_00 = a[0,0] @ b[0,0]
        ab_00_01 = a[0,0] @ b[0,1]
        ab_01_00 = a[0,1] @ b[0,0]
        ab_01_01 = a[0,1] @ b[0,1]
        ab_10_01 = a[1,0] @ b[0,1]

        res_c_00_01 = ab_01_00 @ c[1,0]
        res_c_00 = 2 * ab_00_00 @ c[0,0] \
                + res_c_00_01 \
                + symmetry_c * res_c_00_01.floquetConjugate()
        res_c_01 = 2 * ab_00_01 @ c[0,0] \
                + ab_01_01 @ c[1,0] \
                + ab_10_01 @ c[0,1]
        res_c = ReservoirMatrix(a.global_properties, symmetry_c)
        res_c.voltage_shifts = a.voltage_shifts + b.voltage_shifts + c.voltage_shifts
        res_c[0,0] = res_c_00
        res_c[1,1] = res_c_00.copy()
        res_c[0,1] = res_c_01
        res_c[1,0] = symmetry_c * res_c_01.floquetConjugate()

        res_d_00_01 = ab_01_00 @ d[1,0]
        res_d_00 = 2 * ab_00_00 @ d[0,0] \
                + res_d_00_01 \
                + symmetry_d * res_d_00_01.floquetConjugate()
        res_d_01 = 2 * ab_00_01 @ d[0,0] + ab_01_01 @ d[1,0] + ab_10_01 @ d[0,1]
        res_d = ReservoirMatrix(a.global_properties, symmetry_d)
        res_d.voltage_shifts = a.voltage_shifts + b.voltage_shifts + d.voltage_shifts
        res_d[0,0] = res_d_00
        res_d[1,1] = res_d_00.copy()
        res_d[0,1] = res_d_01
        res_d[1,0] = symmetry_d * res_d_01.floquetConjugate()
        return res_c, res_d
    else:
        # TODO: check!
        # xL != xR
        ab_00_00 = a[0,0] @ b[0,0]
        ab_11_00 = a[1,1] @ b[0,0]
        ab_00_11 = a[0,0] @ b[1,1]
        ab_11_11 = a[1,1] @ b[1,1]
        ab_00_01 = a[0,0] @ b[0,1]
        ab_11_01 = a[1,1] @ b[0,1]
        ab_01_00 = a[0,1] @ b[0,0]
        ab_01_11 = a[0,1] @ b[1,1]
        ab_01_01 = a[0,1] @ b[0,1]
        ab_10_01 = a[1,0] @ b[0,1]

        res_c_00_01 = ab_01_00 @ c[1,0]
        res_c_11_01 = ab_01_11 @ c[1,0]
        res_c_00 = ab_00_00 @ c[0,0] \
                + ab_11_00 @ c[1,1] \
                + res_c_00_01 \
                + symmetry_c * res_c_00_01.floquetConjugate()
        res_c_11 = ab_00_11 @ c[0,0] \
                + ab_11_11 @ c[1,1] \
                + res_c_11_01 \
                + symmetry_c * res_c_11_01.floquetConjugate()
        res_c_01 = ab_11_01 @ c[1,1] \
                + ab_00_01 @ c[0,0] \
                + ab_01_01 @ c[1,0] \
                + ab_10_01 @ c[0,1]
        res_c = ReservoirMatrix(a.global_properties, symmetry_c)
        res_c.voltage_shifts = a.voltage_shifts + b.voltage_shifts + c.voltage_shifts
        res_c[0,0] = res_c_00
        res_c[1,1] = res_c_11
        res_c[0,1] = res_c_01
        res_c[1,0] = symmetry_c * res_c_01.floquetConjugate()

        res_d_00_01 = ab_01_00 @ d[1,0]
        res_d_11_01 = ab_01_11 @ d[1,0]
        res_d_00 = ab_00_00 @ d[0,0] \
                + ab_11_00 @ d[1,1] \
                + res_d_00_01 \
                + symmetry_d * res_d_00_01.floquetConjugate()
        res_d_11 = ab_00_11 @ d[0,0] \
                + ab_11_11 @ d[1,1] \
                + res_d_11_01 \
                + symmetry_d * res_d_11_01.floquetConjugate()
        res_d_01 = ab_11_01 @ d[1,1] \
                + ab_00_01 @ d[0,0] \
                + ab_01_01 @ d[1,0] \
                + ab_10_01 @ d[0,1]
        res_d = ReservoirMatrix(a.global_properties, symmetry_d)
        res_d.voltage_shifts = a.voltage_shifts + b.voltage_shifts + d.voltage_shifts
        res_d[0,0] = res_d_00
        res_d[1,1] = res_d_11
        res_d[0,1] = res_d_01
        res_d[1,0] = symmetry_d * res_d_01.floquetConjugate()
        return res_c, res_d


def product_combinations(
        a: ReservoirMatrix,
        b: ReservoirMatrix
        ) -> (ReservoirMatrix, ReservoirMatrix):
    """
    Equivalent to

        lambda a, b: a @ b, a % b

    but more efficient.
    Arguments must be two ReservoirMatrices.

    12 multiplications (instead of 16) without symmetry,
    1 ReservoirMatrix multiplication with symmetry
    Number of multiplications when using symmetries:
        4 multiplications with xL == xR,
        7 multiplications with xL != xR,
        8 multiplications without symmetry
    """
    assert isinstance(a, ReservoirMatrix)
    assert isinstance(b, ReservoirMatrix)
    assert a.global_properties is b.global_properties
    symmetry = a.symmetry * b.symmetry
    if symmetry == 0 or settings.IGNORE_SYMMETRIES:
        if settings.ENFORCE_SYMMETRIC:
            raise RuntimeError(
                'Unsymmetric product_combinations: %d %d'%(a.symmetry, b.symmetry))
        ab_0000 = a[0,0] @ b[0,0]
        ab_0110 = a[0,1] @ b[1,0]
        ab_1001 = a[1,0] @ b[0,1]
        ab_1111 = a[1,1] @ b[1,1]
        ab = ReservoirMatrix(a.global_properties)
        ab_cross = ReservoirMatrix(a.global_properties)
        ab[0,0] = ab_0000 + ab_0110
        ab[1,1] = ab_1001 + ab_1111
        ab[0,1] = a[0,0] @ b[0,1] + a[0,1] @ b[1,1]
        ab[1,0] = a[1,0] @ b[0,0] + a[1,1] @ b[1,0]
        ab_cross[0,0] = ab_0000 + ab_1001
        ab_cross[1,1] = ab_0110 + ab_1111
        ab_cross[0,1] = a[0,1] @ b[0,0] + a[1,1] @ b[0,1]
        ab_cross[1,0] = a[0,0] @ b[1,0] + a[1,0] @ b[1,1]
        return ab, ab_cross

    ab = a @ b
    ab_cross = ReservoirMatrix(a.global_properties)
    ab_cross[0,0] = symmetry * ab[0,0].floquetConjugate()
    ab_cross[0,1] = symmetry * ab[1,0].floquetConjugate()
    ab_cross[1,0] = symmetry * ab[0,1].floquetConjugate()
    ab_cross[1,1] = symmetry * ab[1,1].floquetConjugate()
    return ab, ab_cross
