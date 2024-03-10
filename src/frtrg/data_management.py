# Copyright 2022 Valentin Bruch <valentin.bruch@rwth-aachen.de>
# License: MIT
"""
Kondo FRTRG, data management module

This file contains functions and classes to manage data generated using the
kondo module.

General concepts:

* All metadata are stored in an SQL database.
* Floquet matrices are stored in HDF5 files.
* Each HDF5 file can contain multiple data points. Data points can be added to
  HDF5 files.
* Each HDF5 file contains a table of metadata for the data points stored in
  this file.
* Data points are identified in HDF5 files by a hash generated from their full
  Floquet matrices at the end of the RG flow.
* The SQL database stores the directory, filename, and hash where the Floquet
  matrices are stored.

Implementation:

* pandas for accessing the SQL database and managing the full table of metadata
* pytables for HDF5 files
* a file "filename.lock" is temporarily created when writing to a HDF5 file.
"""

import os
import tables as tb
import pathlib
from time import sleep
from datetime import datetime
import numpy as np
import pandas as pd
import sqlalchemy as db
import random
import warnings
from . import settings

# We use hashs as identifiers for data points in HDF5 files. These hashs are
# often not valid python names, which causes a warning. We ignore this warning.
warnings.simplefilter("ignore", tb.NaturalNameWarning)


def reduce_h5f(filename):
    """
    Remove voltage branches in all Floquet matrices saved in H5 file.
    CAUTION: This file leads to data loss!

    To really save disk space:
    >>> reduce_h5f(origname)
    >>> with open_file(origname, "r") as h5f:
    >>>     h5f.copy_file(newname)

    and then `cp newname origname`
    """
    with tb.open_file(filename, "a") as h5f:
        nodes = h5f.list_nodes("/data")
        for group in nodes:
            name = group._v_name
            for var in ("gamma", "z", "deltaGamma"):
                node = group[var]
                if (
                    node.ndim == 3
                    and node.shape[0] % 2
                    and node.shape[1] == node.shape[2]
                ):
                    vb = node.shape[0] // 2
                    arr = node[vb].copy()
                    h5f.remove_node(f"/data/{name}", var)
                    h5f.create_array(f"/data/{name}", var, arr)
            for var in ("g2", "g3"):
                node = group[var]
                if (
                    node.ndim == 5
                    and node.shape[0] == 2
                    and node.shape[1] == 2
                    and node.shape[2] % 2
                    and node.shape[3] == node.shape[4]
                ):
                    vb = node.shape[2] // 2
                    arr = node[:, :, vb].copy()
                    h5f.remove_node(f"/data/{name}", var)
                    h5f.create_array(f"/data/{name}", var, arr)


def random_string(length: int):
    """
    Generate random strings of alphanumerical characters with given length.
    """
    res = ""
    for _ in range(length):
        x = random.randint(0, 61)
        if x < 10:
            res += chr(x + 48)
        elif x < 36:
            res += chr(x + 55)
        else:
            res += chr(x + 61)
    return res


def replace_all(string: str, replacements: dict):
    """
    Apply all replacements to string
    """
    for old, new in replacements.items():
        string = string.replace(old, new)
    return string


class KondoExport:
    """
    Class for saving Kondo object to file.
    Example usage:
    >>> kondo = Kondo(...)
    >>> kondo.run(...)
    >>> KondoExport(kondo).save_h5("data/frtrg-01.h5")
    """

    METHOD_ENUM = tb.Enum(
        (
            "unknown",
            "mu",
            "J",
            "J-compact-1",
            "J-compact-2",
            "mu-reference",
            "J-reference",
            "mu-extrap-voltage",
            "J-extrap-voltage",
        )
    )
    SOLVER_METHOD_ENUM = tb.Enum(
        ("unknown", "RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA", "other")
    )

    def __init__(self, kondo):
        self.kondo = kondo

    @property
    def hash(self):
        """
        hash based on Floquet matrices in Kondo object
        """
        try:
            return self._hash
        except AttributeError:
            self._hash = self.kondo.hash()[:40]
            return self._hash

    @property
    def metadata(self):
        """
        dictionary of metadata
        """
        # Determine method
        if self.kondo.unitary_transformation:
            if self.kondo.compact == 2:
                method = "J-compact-2"
            elif self.kondo.compact == 1:
                method = "J-compact-1"
            else:
                method = "J"
        else:
            method = "mu"

        # Collect solver flags
        solver_flags = 0
        try:
            if self.kondo.simplified_initial_conditions:
                solver_flags |= DataManager.SOLVER_FLAGS[
                    "simplified_initial_conditions"
                ]
        except AttributeError:
            pass
        try:
            if self.kondo.improved_initial_conditions:
                solver_flags |= DataManager.SOLVER_FLAGS["improved_initial_conditions"]
        except AttributeError:
            pass
        try:
            if self.kondo.include_Ga:
                solver_flags |= DataManager.SOLVER_FLAGS["include_Ga"]
        except AttributeError:
            pass
        try:
            if self.kondo.solve_integral_exactly:
                solver_flags |= DataManager.SOLVER_FLAGS["solve_integral_exactly"]
        except AttributeError:
            pass
        try:
            if self.kondo.truncation_order == 2:
                solver_flags |= DataManager.SOLVER_FLAGS["second_order_rg_equations"]
            elif self.kondo.truncation_order != 3:
                settings.logger.warning(
                    "Invalid truncation order: %s" % self.kondo.truncation_order
                )
        except AttributeError:
            pass
        for key, value in self.kondo.global_settings.items():
            if value:
                try:
                    solver_flags |= DataManager.SOLVER_FLAGS[key.lower()]
                except KeyError:
                    pass

        version = self.kondo.global_settings["VERSION"]
        return dict(
            hash=self.hash,
            omega=self.kondo.omega,
            energy=self.kondo.energy,
            version_major=int(version[0]),
            version_minor=int(version[1]),
            lazy_inverse_factor=self.kondo.global_settings["LAZY_INVERSE_FACTOR"],
            git_commit_count=version[2],
            git_commit_id=version[3],
            method=method,
            timestamp=datetime.utcnow().timestamp(),
            solver_method=getattr(self.kondo, "solveopts", {}).get("method", "unknown"),
            solver_tol_abs=getattr(self.kondo, "solveopts", {}).get("atol", -1),
            solver_tol_rel=getattr(self.kondo, "solveopts", {}).get("rtol", -1),
            integral_method=getattr(self.kondo, "integral_method", -15),
            d=self.kondo.d,
            vdc=self.kondo.vdc,
            vac=self.kondo.vac,
            xL=self.kondo.xL,
            nmax=int(self.kondo.nmax),
            padding=int(self.kondo.padding),
            voltage_branches=int(self.kondo.voltage_branches),
            resonant_dc_shift=int(self.kondo.resonant_dc_shift),
            solver_flags=int(solver_flags),
        )

    @property
    def main_results(self):
        """
        dictionary of main results: DC current, DC conductance, AC current (absolute value and phase)
        """
        results = dict(
            gamma=np.nan,
            dc_current=np.nan,
            dc_conductance=np.nan,
            ac_current_abs=np.nan,
            ac_current_phase=np.nan,
            pole=getattr(self.kondo, "pole", np.nan),
        )
        nmax = self.kondo.nmax
        vb = self.kondo.voltage_branches
        if self.kondo.compact == 0:
            try:
                results["gamma"] = self.kondo.gamma[vb, nmax, nmax].real
            except:
                pass
            try:
                results["dc_current"] = self.kondo.gammaL[nmax, nmax].real
            except:
                pass
            try:
                results["dc_conductance"] = self.kondo.deltaGammaL[nmax, nmax].real
            except:
                pass
            if nmax == 0:
                results["ac_current_abs"] = 0
            else:
                try:
                    results["ac_current_abs"] = np.abs(
                        self.kondo.gammaL[nmax - 1, nmax]
                    )
                    results["ac_current_phase"] = np.angle(
                        self.kondo.gammaL[nmax - 1, nmax]
                    )
                except:
                    pass
        elif self.kondo.compact:
            results["dc_current"] = 0
            if nmax % 2:
                try:
                    results["gamma"] = self.kondo.gamma.submatrix11[
                        nmax // 2, nmax // 2
                    ].real
                except:
                    pass
                try:
                    results["dc_conductance"] = self.kondo.deltaGammaL.submatrix11[
                        nmax // 2, nmax // 2
                    ].real
                except:
                    pass
                try:
                    results["ac_current_abs"] = np.abs(
                        self.kondo.gammaL.submatrix01[nmax // 2, nmax // 2]
                    )
                    results["ac_current_phase"] = np.angle(
                        self.kondo.gammaL.submatrix01[nmax // 2, nmax // 2]
                    )
                except:
                    pass
            else:
                try:
                    results["gamma"] = self.kondo.gamma.submatrix00[
                        nmax // 2, nmax // 2
                    ].real
                except:
                    pass
                try:
                    results["dc_conductance"] = self.kondo.deltaGammaL.submatrix00[
                        nmax // 2, nmax // 2
                    ].real
                except:
                    pass
                try:
                    results["ac_current_abs"] = np.abs(
                        self.kondo.gammaL.submatrix10[nmax // 2 - 1, nmax // 2]
                    )
                    results["ac_current_phase"] = np.angle(
                        self.kondo.gammaL.submatrix10[nmax // 2 - 1, nmax // 2]
                    )
                except:
                    pass
        return results

    @property
    def fourier_coef(self):
        return self.kondo.fourier_coef

    def data(self, include="all"):
        """
        Return Floquet matrices in a dictionary of numpy array

        Parameters
        ----------
        include : {"all", "reduced", "observables", "minimal"}
            "all"
                save all data (Floquet matrices including voltage shifts)
            "reduced"
                exclude voltage shifts and yL
            "observables
                save only gamma, gammaL, deltaGammaL, excluding voltage
                shifts
            "minimal"
                save only central column of Floquet matrices for gamma,
                gammaL, deltaGammaL, excluding voltage

        Returns
        -------
        dict
            dictionary of Floquet matrices as numpy arrays.
        """
        if include == "all":
            save = dict(
                gamma=self.kondo.gamma.values,
                z=self.kondo.z.values,
                gammaL=self.kondo.gammaL.values,
                deltaGammaL=self.kondo.deltaGammaL.values,
                deltaGamma=self.kondo.deltaGamma.values,
                yL=self.kondo.yL.values,
                g2=self.kondo.g2.to_numpy_array(),
                g3=self.kondo.g3.to_numpy_array(),
                current=self.kondo.current.to_numpy_array(),
            )
            if self.kondo.include_Ga:
                try:
                    save["ga_scalar"] = self.kondo.ga_scalar.values
                except AttributeError:
                    save["ga"] = self.kondo.ga.to_numpy_array()
        elif include == "reduced":
            if self.kondo.voltage_branches:
                vb = self.kondo.voltage_branches
                save = dict(
                    gamma=self.kondo.gamma[vb],
                    z=self.kondo.z[vb],
                    gammaL=self.kondo.gammaL.values,
                    deltaGammaL=self.kondo.deltaGammaL.values,
                    deltaGamma=self.kondo.deltaGamma[min(vb, 1)],
                    g2=self.kondo.g2.to_numpy_array()[:, :, vb],
                    g3=self.kondo.g3.to_numpy_array()[:, :, vb],
                    current=self.kondo.current.to_numpy_array(),
                )
                if self.kondo.include_Ga:
                    try:
                        save["ga_scalar"] = self.kondo.ga_scalar[vb]
                    except AttributeError:
                        save["ga"] = self.kondo.ga.to_numpy_array()[:, :, vb]
            else:
                save = dict(
                    gamma=self.kondo.gamma.values,
                    z=self.kondo.z.values,
                    gammaL=self.kondo.gammaL.values,
                    deltaGammaL=self.kondo.deltaGammaL.values,
                    deltaGamma=self.kondo.deltaGamma.values,
                    g2=self.kondo.g2.to_numpy_array(),
                    g3=self.kondo.g3.to_numpy_array(),
                    current=self.kondo.current.to_numpy_array(),
                )
                if self.kondo.include_Ga:
                    try:
                        save["ga_scalar"] = self.kondo.ga_scalar.values
                    except AttributeError:
                        save["ga"] = self.kondo.ga.to_numpy_array()
        elif include == "observables":
            if self.kondo.voltage_branches:
                vb = self.kondo.voltage_branches
                save = dict(
                    gamma=self.kondo.gamma[vb],
                    gammaL=self.kondo.gammaL.values,
                    deltaGammaL=self.kondo.deltaGammaL.values,
                )
            else:
                save = dict(
                    gamma=self.kondo.gamma.values,
                    gammaL=self.kondo.gammaL.values,
                    deltaGammaL=self.kondo.deltaGammaL.values,
                )
        elif include == "minimal":
            nmax = self.kondo.nmax
            if self.kondo.voltage_branches:
                vb = self.kondo.voltage_branches
                save = dict(
                    gamma=self.kondo.gamma[vb, :, nmax],
                    gammaL=self.kondo.gammaL[:, nmax],
                    deltaGammaL=self.kondo.deltaGammaL[:, nmax],
                )
            else:
                save = dict(
                    gamma=self.kondo.gamma[:, nmax],
                    gammaL=self.kondo.gammaL[:, nmax],
                    deltaGammaL=self.kondo.deltaGammaL[:, nmax],
                )
        else:
            raise ValueError("Unknown value for include: " + include)
        return save

    def save_npz(self, filename, include="all"):
        """
        Save data in binary numpy format.
        """
        np.savez(filename, **self.metadata, **self.data(include))

    def save_h5(self, filename, include="all", overwrite=False):
        """
        Save data in HDF5 file.

        Returns absolute path to filename where data have been saved.
        If overwrite is False and a file would be overwritten, append a random
        string to the end of the filename.
        """
        while True:
            try:
                pathlib.Path(filename + ".lock").touch(exist_ok=False)
                break
            except FileExistsError:
                try:
                    settings.logger.warning(
                        "File %s is locked, waiting ~0.5s" % filename
                    )
                    sleep(0.4 + 0.2 * random.random())
                except KeyboardInterrupt:
                    answer = input('Ignore lock file? Then type "yes": ')
                    if answer.lower() == "yes":
                        break
                    answer = input(
                        "Save with filename extended by random string? (Yn): "
                    )
                    if answer.lower()[0] != "n":
                        return self.save_h5(
                            filename.removesuffix(".h5") + random_string(8) + ".h5",
                            include,
                            overwrite,
                        )
        try:
            file_exists = os.path.exists(filename)
            h5file = None
            while h5file is None:
                try:
                    h5file = tb.open_file(
                        filename, "a", MAX_NUMEXPR_THREADS=1, MAX_BLOSC_THREADS=1
                    )
                except tb.exceptions.HDF5ExtError:
                    settings.logger.warning(
                        "Error opening file %s, waiting 0.5s" % filename
                    )
                    sleep(0.5)
            try:
                if file_exists:
                    try:
                        h5file.is_visible_node("/data/" + self.hash)
                        new_filename = (
                            filename.removesuffix(".h5") + random_string(8) + ".h5"
                        )
                        settings.logger.warning(
                            "Hash exists in file %s! Saving to %s"
                            % (filename, new_filename)
                        )
                        return self.save_h5(new_filename, include, overwrite)
                    except tb.exceptions.NoSuchNodeError:
                        pass
                    metadata_table = h5file.get_node("/metadata/mdtable")
                else:
                    # create new file
                    metadata_parent = h5file.create_group(
                        h5file.root, "metadata", "Metadata"
                    )
                    metadata_table = h5file.create_table(
                        metadata_parent,
                        "mdtable",
                        dict(
                            idnum=tb.Int32Col(),
                            hash=tb.StringCol(40),
                            omega=tb.Float64Col(),
                            energy=tb.ComplexCol(16),
                            version_major=tb.Int16Col(),
                            version_minor=tb.Int16Col(),
                            git_commit_count=tb.Int16Col(),
                            git_commit_id=tb.Int32Col(),
                            timestamp=tb.Time64Col(),
                            method=tb.EnumCol(
                                KondoExport.METHOD_ENUM, "unknown", "int8"
                            ),
                            solver_method=tb.EnumCol(
                                KondoExport.SOLVER_METHOD_ENUM, "unknown", "int8"
                            ),
                            solver_tol_abs=tb.Float64Col(),
                            solver_tol_rel=tb.Float64Col(),
                            integral_method=tb.Int16Col(),
                            d=tb.Float64Col(),
                            vdc=tb.Float64Col(),
                            vac=tb.Float64Col(),
                            xL=tb.Float64Col(),
                            nmax=tb.Int16Col(),
                            padding=tb.Int16Col(),
                            voltage_branches=tb.Int16Col(),
                            resonant_dc_shift=tb.Int16Col(),
                            solver_flags=tb.Int16Col(),
                            lazy_inverse_factor=tb.Float64Col(),
                        ),
                    )
                    h5file.create_group(h5file.root, "data", "Floquet matrices")
                    h5file.flush()

                # Save metadata
                row = metadata_table.row
                idnum = metadata_table.shape[0]
                row["idnum"] = idnum
                try:
                    if include != "all":
                        self.metadata["solver_flags"] |= DataManager.SOLVER_FLAGS[
                            "reduced"
                        ]
                except:
                    settings.logger.exception("Error while updating solver flags")
                metadata = self.metadata
                row["method"] = KondoExport.METHOD_ENUM[metadata.pop("method")]
                row["solver_method"] = KondoExport.SOLVER_METHOD_ENUM[
                    metadata.pop("solver_method")
                ]
                for key, value in metadata.items():
                    try:
                        row[key] = value
                    except KeyError:
                        settings.logger.warning(
                            "failed to set metadata value:", exc_info=1
                        )
                row.append()

                # save data
                datagroup = h5file.create_group("/data/", self.hash)
                data = self.data(include)
                for key, value in data.items():
                    h5file.create_array(datagroup, key, value)
                if self.fourier_coef is not None:
                    h5file.create_array(
                        datagroup, "fourier_coef", np.array(self.fourier_coef)
                    )
                h5file.flush()
            finally:
                h5file.close()
        finally:
            os.remove(filename + ".lock")
        return os.path.abspath(filename)


class KondoImport:
    """
    Class for importing Kondo objects that were saved with KondoExport.
    Example usage:
    >>> kondo, = KondoImport.read_from_h5("data/frtrg-01.h5", "94f81d2b49df15912798d95cae8e108d75c637c2")
    >>> print(kondo.gammaL[kondo.nmax, kondo.nmax])
    """

    def __init__(self, metadata, datanode, h5file, owns_h5file=False):
        self.metadata = metadata
        self._datanode = datanode
        self._h5file = h5file
        self._owns_h5file = owns_h5file

    def __del__(self):
        if self._owns_h5file:
            settings.logger.info("closing h5file")
            self._h5file.close()

    @classmethod
    def read_from_h5(cls, filename, khash):
        h5file = tb.open_file(filename, "r")
        datanode = h5file.get_node("/data/" + khash)
        metadatatable = h5file.get_node("/metadata/mdtable")
        counter = 0
        for row in metadatatable.where(f"hash == '{khash}'"):
            metadata = {key: row[key] for key in metadatatable.colnames}
            metadata.pop("idnum", None)
            metadata["hash"] = row["hash"].decode()
            metadata["method"] = KondoExport.METHOD_ENUM(row["method"])
            metadata["solver_method"] = KondoExport.SOLVER_METHOD_ENUM(
                row["solver_method"]
            )
            item = cls(metadata, datanode, h5file)
            item._rawmetadata = row
            yield item
            counter += 1
        if counter == 1:
            item._owns_h5file = True
        else:
            settings.logger.warning("h5file will not be closed automatically")

    @classmethod
    def read_all_from_h5(cls, filename):
        h5file = tb.open_file(filename)
        metadatatable = h5file.get_node("/metadata/mdtable")
        counter = 0
        for row in metadatatable:
            metadata = {key: row[key] for key in metadatatable.colnames}
            metadata.pop("idnum", None)
            metadata["hash"] = row["hash"].decode()
            metadata["method"] = KondoExport.METHOD_ENUM(row["method"])
            metadata["solver_method"] = KondoExport.SOLVER_METHOD_ENUM(
                row["solver_method"]
            )
            datanode = h5file.get_node("/data/" + metadata["hash"])
            item = cls(metadata, datanode, h5file)
            item._rawmetadata = row
            yield item
            counter += 1
        if counter == 1:
            item._owns_h5file = True
        else:
            settings.logger.warning("h5file will not be closed automatically")

    @property
    def main_results(self):
        """
        dictionary of main results: DC current, DC conductance, AC current (absolute value and phase)
        """
        results = dict(
            gamma=np.nan,
            dc_current=np.nan,
            dc_conductance=np.nan,
            ac_current_abs=np.nan,
            ac_current_phase=np.nan,
        )
        nmax = self.nmax
        if self.method in (
            "unknown",
            "mu",
            "J",
            "mu-reference",
            "J-reference",
            "mu-extrap-voltage",
            "J-extrap-voltage",
        ):
            voltage_branches = self.voltage_branches
            try:
                results["dc_current"] = self.gammaL[nmax, nmax].real
            except:
                pass
            try:
                results["dc_conductance"] = self.deltaGammaL[nmax, nmax].real
            except:
                pass
            try:
                gamma = self._datanode["gamma"].read()
                if gamma.ndim == 3:
                    results["gamma"] = gamma[voltage_branches, nmax, nmax].real
                elif gamma.ndim == 2:
                    results["gamma"] = gamma[nmax, nmax].real
                else:
                    results["gamma"] = gamma[nmax].real
            except:
                pass
            if nmax == 0:
                results["ac_current_abs"] = 0
            else:
                try:
                    results["ac_current_abs"] = np.abs(self.gammaL[nmax - 1, nmax])
                    results["ac_current_phase"] = np.angle(self.gammaL[nmax - 1, nmax])
                except:
                    pass
        elif self.method in ("J-compact-1", "J-compact-2"):
            results["dc_current"] = 0
            if nmax % 2:
                try:
                    results["gamma"] = self.gamma.submatrix11[nmax // 2, nmax // 2].real
                except:
                    pass
                try:
                    results["dc_conductance"] = self.deltaGammaL.submatrix11[
                        nmax // 2, nmax // 2
                    ].real
                except:
                    pass
                try:
                    results["ac_current_abs"] = np.abs(
                        self.gammaL.submatrix01[nmax // 2, nmax // 2]
                    )
                    results["ac_current_phase"] = np.angle(
                        self.gammaL.submatrix01[nmax // 2, nmax // 2]
                    )
                except:
                    pass
            else:
                try:
                    results["gamma"] = self.gamma.submatrix00[nmax // 2, nmax // 2].real
                except:
                    pass
                try:
                    results["dc_conductance"] = self.deltaGammaL.submatrix00[
                        nmax // 2, nmax // 2
                    ].real
                except:
                    pass
                try:
                    results["ac_current_abs"] = np.abs(
                        self.gammaL.submatrix10[nmax // 2 - 1, nmax // 2]
                    )
                    results["ac_current_phase"] = np.angle(
                        self.gammaL.submatrix10[nmax // 2 - 1, nmax // 2]
                    )
                except:
                    pass
        return results

    @property
    def fourier_coef(self):
        if "fourier_coef" in self._datanode:
            return self._datanode.fourier_coef.read()
        return None

    def __getitem__(self, name):
        if name in self.metadata:
            return self.metadata[name]
        if name in self._datanode:
            return self._datanode[name].read()
        raise KeyError("Unknown key: %s" % name)

    def __getattr__(self, name):
        if name in self._datanode:
            return self._datanode[name].read()
        if name in self.metadata:
            return self.metadata[name]
        raise AttributeError("Unknown attribute name: %s" % name)


class DataManager:
    """
    Interface to database using pytables

    tables:
        datapoints (single data point)
    """

    SOLVER_FLAGS = dict(
        contains_flow=0x0001,
        reduced=0x0002,
        deleted=0x0004,
        simplified_initial_conditions=0x0008,
        enforce_symmetric=0x0010,
        check_symmetries=0x0020,
        ignore_symmetries=0x0040,
        extrapolate_voltage=0x0080,
        use_cublas=0x0100,
        use_reference_implementation=0x0200,
        second_order_rg_equations=0x0400,
        solve_integral_exactly=0x0800,
        include_Ga=0x1000,
        improved_initial_conditions=0x2000,
    )

    def __init__(self):
        self.version = settings.VERSION
        self.engine = db.create_engine(
            settings.DB_CONNECTION_STRING, future=True, echo=False
        )

        self.metadata = db.MetaData()
        try:
            self.table = db.Table(
                "datapoints", self.metadata, autoload=True, autoload_with=self.engine
            )
        except db.exc.NoSuchTableError:
            with self.engine.begin() as connection:
                settings.logger.info("Creating database table datapoints")
                self.table = db.Table(
                    "datapoints",
                    self.metadata,
                    db.Column("id", db.INTEGER(), primary_key=True),
                    db.Column("hash", db.CHAR(40)),
                    db.Column("version_major", db.SMALLINT()),
                    db.Column("version_minor", db.SMALLINT()),
                    db.Column("git_commit_count", db.SMALLINT()),
                    db.Column("git_commit_id", db.INTEGER()),
                    db.Column("timestamp", db.TIMESTAMP()),
                    db.Column(
                        "method",
                        db.Enum(
                            "unknown",
                            "mu",
                            "J",
                            "J-compact-1",
                            "J-compact-2",
                            "mu-reference",
                            "J-reference",
                        ),
                    ),
                    db.Column(
                        "solver_method",
                        db.Enum(
                            "unknown",
                            "RK45",
                            "RK23",
                            "DOP853",
                            "Radau",
                            "BDF",
                            "LSODA",
                            "other",
                        ),
                    ),
                    db.Column("solver_tol_abs", db.FLOAT()),
                    db.Column("solver_tol_rel", db.FLOAT()),
                    db.Column("omega", db.FLOAT()),
                    db.Column("d", db.FLOAT()),
                    db.Column("vdc", db.FLOAT()),
                    db.Column("vac", db.FLOAT()),
                    db.Column("xL", db.FLOAT()),
                    db.Column("energy_re", db.FLOAT()),
                    db.Column("energy_im", db.FLOAT()),
                    db.Column("lazy_inverse_factor", db.FLOAT()),
                    db.Column("dc_current", db.FLOAT()),
                    db.Column("ac_current_abs", db.FLOAT()),
                    db.Column("ac_current_phase", db.FLOAT()),
                    db.Column("dc_conductance", db.FLOAT()),
                    db.Column("gamma", db.FLOAT()),
                    db.Column("pole", db.FLOAT()),
                    db.Column("nmax", db.SMALLINT()),
                    db.Column("padding", db.SMALLINT()),
                    db.Column("voltage_branches", db.SMALLINT()),
                    db.Column("resonant_dc_shift", db.SMALLINT()),
                    db.Column("solver_flags", db.SMALLINT()),
                    db.Column("integral_method", db.SMALLINT()),
                    db.Column("dirname", db.String(256)),
                    db.Column("basename", db.String(128)),
                    db.Column("fourier_coef_id", db.INTEGER(), default=-1),
                )
                self.table.create(bind=connection)
        try:
            self.fourier_coef_table = db.Table(
                "fourier_coef", self.metadata, autoload=True, autoload_with=self.engine
            )
        except db.exc.NoSuchTableError:
            with self.engine.begin() as connection:
                settings.logger.info("Creating database table fourier_coef")
                self.fourier_coef_table = db.Table(
                    # NOTE: wrong indices!!
                    "fourier_coef",
                    self.metadata,
                    db.Column("id", db.INTEGER(), primary_key=True),
                    db.Column("fcr0", db.FLOAT(), default=0.0),
                    db.Column("fci0", db.FLOAT(), default=0.0),
                    db.Column("fcr1", db.FLOAT(), default=0.0),
                    db.Column("fci1", db.FLOAT(), default=0.0),
                    db.Column("fcr2", db.FLOAT(), default=0.0),
                    db.Column("fci2", db.FLOAT(), default=0.0),
                    db.Column("fcr3", db.FLOAT(), default=0.0),
                    db.Column("fci3", db.FLOAT(), default=0.0),
                    db.Column("fcr4", db.FLOAT(), default=0.0),
                    db.Column("fci4", db.FLOAT(), default=0.0),
                    db.Column("fcr5", db.FLOAT(), default=0.0),
                    db.Column("fci5", db.FLOAT(), default=0.0),
                    db.Column("fcr6", db.FLOAT(), default=0.0),
                    db.Column("fci6", db.FLOAT(), default=0.0),
                    db.Column("fcr7", db.FLOAT(), default=0.0),
                    db.Column("fci7", db.FLOAT(), default=0.0),
                    db.Column("fcr8", db.FLOAT(), default=0.0),
                    db.Column("fci8", db.FLOAT(), default=0.0),
                    db.Column("fcr9", db.FLOAT(), default=0.0),
                    db.Column("fci9", db.FLOAT(), default=0.0),
                )
                self.fourier_coef_table.create(bind=connection)

    def insert_from_h5file(self, filename):
        """
        Scan data in HDF5 file and insert datasets in database if they are not
        included yet.
        """
        filename = os.path.abspath(filename)
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename)
        datasets = []
        skipped = 0
        for dataset in KondoImport.read_all_from_h5(filename):
            settings.logger.debug("Checking hash=" + dataset.hash)
            candidates = self.df_table.loc[self.df_table.hash == dataset.hash]
            if candidates.shape[0] > 0:
                settings.logger.debug(
                    f"Found {candidates.shape[0]} times the same hash"
                )
                exists = False
                for idx, candidate in candidates.iterrows():
                    if os.path.join(candidate.dirname, candidate.basename) == filename:
                        exists = True
                        break
                if exists:
                    settings.logger.debug("Entry exists, skipping")
                    skipped += 1
                    continue
                else:
                    settings.logger.debug("File seems new, continuing anyway")
            metadata = dataset.metadata.copy()
            energy = metadata.pop("energy")
            metadata.update(
                energy_re=energy.real,
                energy_im=energy.imag,
                timestamp=datetime.fromtimestamp(metadata.pop("timestamp"))
                .isoformat()
                .replace("T", " "),
                dirname=dirname,
                basename=basename,
                fourier_coef_id=-1,
            )
            metadata.update(dataset.main_results)
            fourier_coef = dataset.fourier_coef
            if fourier_coef is not None:
                values = {
                    "fcr%d" % i: f for i, f in enumerate(dataset.fourier_coef[:10].real)
                }
                values.update(
                    {
                        "fci%d" % i: f
                        for i, f in enumerate(dataset.fourier_coef[:10].imag)
                    }
                )
                stmt = self.fourier_coef_table.insert(values=values)
                with self.engine.begin() as connection:
                    result = connection.execute(stmt)
                    connection.commit()
                metadata.update(fourier_coef_id=result.inserted_primary_key[0])
            datasets.append(metadata)
        try:
            if not dataset._owns_h5file:
                dataset._h5file.close()
                settings.logger.info("Closed HDF5 file")
        except:
            pass
        settings.logger.info(
            f"Inserting {len(datasets)} new entries, ignoring {skipped}"
        )
        new_frame = pd.DataFrame(datasets)
        new_frame.to_sql(
            "datapoints",
            self.engine,
            if_exists="append",
            index=False,
        )
        try:
            del self._df_table
        except AttributeError:
            pass

    def insert_in_db(self, filename: str, kondo: KondoExport):
        """
        Save metadata in database for data stored in filename.
        Returns ID of inserted row.
        """
        metadata = kondo.metadata
        metadata.update(kondo.main_results)
        energy = metadata.pop("energy")
        metadata.update(
            energy_re=energy.real,
            energy_im=energy.imag,
            timestamp=datetime.fromtimestamp(metadata.pop("timestamp")),
            dirname=os.path.dirname(filename),
            basename=os.path.basename(filename),
        )
        try:
            if metadata["pole"] == np.nan:
                metadata.pop("pole")
        except KeyError:
            pass
        fourier_coef = kondo.fourier_coef
        if fourier_coef is not None:
            fourier_coef = np.asarray(fourier_coef, dtype=np.complex128)
            values = {f"fcr{i}": float(f) for i, f in enumerate(fourier_coef[:10].real)}
            values.update(
                {f"fci{i}": float(f) for i, f in enumerate(fourier_coef[:10].imag)}
            )
            stmt = self.fourier_coef_table.insert(values=values)
            with self.engine.begin() as connection:
                result = connection.execute(stmt)
                connection.commit()
            (fourier_coef_id,) = result.inserted_primary_key
            metadata.update(fourier_coef_id=fourier_coef_id)
        stmt = self.table.insert().values(metadata)
        with self.engine.begin() as connection:
            result = connection.execute(stmt)
            connection.commit()
        try:
            del self._df_table
        except AttributeError:
            pass
        return result.inserted_primary_key[0]

    def save_h5(
        self, kondo: KondoExport, filename: str = None, include="all", overwrite=False
    ):
        """
        Save all data in given filename and keep metadata in database.
        Returns ID and hash of inserted row.
        """
        if filename is None:
            filename = os.path.join(settings.BASEPATH, settings.FILENAME)
        if not isinstance(kondo, KondoExport):
            kondo = KondoExport(kondo)
        filename = kondo.save_h5(filename, include, overwrite)
        rowid = self.insert_in_db(filename, kondo)
        return rowid, kondo.hash

    @property
    def df_table(self):
        try:
            return self._df_table
        except AttributeError:
            min_version = settings.MIN_VERSION
            settings.logger.debug("DataManager: cache df_table")
            with self.engine.begin() as connection:
                df_table = pd.read_sql_table("datapoints", connection, index_col="id")
            selection = (
                df_table.solver_flags & DataManager.SOLVER_FLAGS["deleted"]
            ) == 0
            selection &= (df_table.version_major > min_version[0]) | (
                (df_table.version_major == min_version[0])
                & (df_table.version_minor >= min_version[1])
            )
            selection &= df_table.energy_re == 0
            selection &= df_table.energy_im == 0
            if len(min_version) > 2 and min_version[2] > 0:
                selection &= df_table.git_commit_count >= min_version[2]
            self._df_table = df_table[selection]
            return self._df_table

    def list(self, min_version=(14, 0, -1, -1), has_fourier_coef=False, **parameters):
        """
        Print and return DataFrame with selection of physical parameters.
        """
        selection = (self.df_table.version_major > min_version[0]) | (
            self.df_table.version_major == min_version[0]
        ) & (self.df_table.version_minor >= min_version[1])
        selection &= self.df_table.energy_re == 0
        selection &= self.df_table.energy_im == 0
        fourier_coef = parameters.pop("fourier_coef", None)
        if has_fourier_coef or fourier_coef is not None:
            selection &= self.df_table.fourier_coef_id > 0
        elif has_fourier_coef == False:
            selection &= self.df_table.fourier_coef_id < 0
        good_flags = parameters.pop("good_flags", 0)
        if good_flags:
            selection &= self.df_table.solver_flags & good_flags == good_flags
        bad_flags = parameters.pop("bad_flags", 0)
        if bad_flags:
            selection &= self.df_table.solver_flags & bad_flags == 0
        include_Ga = parameters.pop("include_Ga", None)
        if include_Ga:
            selection &= (
                self.df_table.solver_flags & DataManager.SOLVER_FLAGS["include_Ga"]
            ) != 0
        elif include_Ga == False:
            selection &= (
                self.df_table.solver_flags & DataManager.SOLVER_FLAGS["include_Ga"]
            ) == 0
        solve_integral_exactly = parameters.pop("solve_integral_exactly", None)
        if solve_integral_exactly:
            selection &= (
                self.df_table.solver_flags
                & DataManager.SOLVER_FLAGS["solve_integral_exactly"]
            ) != 0
        elif solve_integral_exactly == False:
            selection &= (
                self.df_table.solver_flags
                & DataManager.SOLVER_FLAGS["solve_integral_exactly"]
            ) == 0
        truncation_order = parameters.pop("truncation_order", None)
        if truncation_order == 2:
            selection &= (
                self.df_table.solver_flags
                & DataManager.SOLVER_FLAGS["second_order_rg_equations"]
            ) != 0
        elif truncation_order == 3:
            selection &= (
                self.df_table.solver_flags
                & DataManager.SOLVER_FLAGS["second_order_rg_equations"]
            ) == 0
        if len(min_version) > 2 and min_version[2] > 0:
            selection &= self.df_table.git_commit_count >= min_version[2]
        method = parameters.pop("method", None)
        if method == "J":
            selection &= self.df_table.method != "mu"
        elif method is not None:
            selection &= self.df_table.method == method
        for key, value in parameters.items():
            if value is None:
                continue
            try:
                selection &= np.isclose(
                    self.df_table[key], value, rtol=1e-6, atol=1e-15
                )
            except TypeError:
                try:
                    selection &= self.df_table[key] == value
                except KeyError:
                    settings.logger.warning("Unknown key: %s" % key)

        if fourier_coef is not None:
            if len(fourier_coef) < 10:
                fourier_coef = (
                    *fourier_coef,
                    *(0 for i in range(10 - len(fourier_coef))),
                )
            else:
                fourier_coef = fourier_coef[:10]
            preselected_table = self.df_table.loc[selection]
            with self.engine.begin() as connection:
                df_fourier_coef_table = pd.read_sql_table(
                    "fourier_coef", connection, index_col="id"
                )
            fourier_coef_table = df_fourier_coef_table.loc[
                preselected_table.fourier_coef_id
            ]
            valid = np.ones(fourier_coef_table.shape[0], dtype=bool)
            for i, f in enumerate(fourier_coef):
                valid &= np.isclose(fourier_coef_table[f"fcr{i}"], f.real) & np.isclose(
                    fourier_coef_table[f"fci{i}"], f.imag
                )
            return preselected_table[valid]
        if selection is True:
            result = self.df_table
        else:
            result = self.df_table.loc[selection]
        return result

    def list_fourier_coef(self, **parameters):
        result = self.list(has_fourier_coef=True, **parameters)
        with self.engine.begin() as connection:
            fourier_coef_table = pd.read_sql_table(
                "fourier_coef", connection, index_col="id"
            )
        return result.join(fourier_coef_table, on="fourier_coef_id", how="left")

    def load_all(self, fourier_coef=None, **parameters):
        table = self.list(**parameters)
        if fourier_coef is not None:
            with self.engine.begin() as connection:
                df_fourier_coef_table = pd.read_sql_table(
                    "fourier_coef", connection, index_col="id"
                )
            fourier_coef_table = df_fourier_coef_table.loc[table.fourier_coef_id]
            valid = np.ones(fourier_coef_table.shape[0], dtype=bool)
            for i, f in enumerate(fourier_coef[:10]):
                valid &= np.isclose(
                    fourier_coef_table["fcr%d" % i], f.real
                ) & np.isclose(fourier_coef_table["fci%d" % i], f.imag)
            table = table.loc[valid]
        for (dirname, basename), subtable in table.groupby(["dirname", "basename"]):
            try:
                h5file = tb.open_file(os.path.join(dirname, basename))
            except:
                settings.logger.exception("Error while loading HDF5 file")
                continue
            # metadatatable = h5file.get_node('/metadata/mdtable')
            for index, row in subtable.iterrows():
                try:
                    datanode = h5file.get_node("/data/" + row.hash)
                    # metadatarow = metadatatable.where("hash == '%s'"%(row.hash))
                    yield KondoImport(
                        row, datanode, h5file, owns_h5file=subtable.shape[0] == 1
                    )
                except:
                    settings.logger.exception("Error while loading data")

    def save_pole(self, kid, khash, pole):
        stmt = (
            self.table.update()
            .where((self.table.c.id == kid) & (self.table.c.hash == khash))
            .values(pole=pole)
        )
        with self.engine.begin() as connection:
            result = connection.execute(stmt)
            connection.commit()


def list_data(**kwargs):
    table = DataManager().list(**kwargs)
    print(
        result[
            [
                "method",
                "vdc",
                "vac",
                "omega",
                "nmax",
                "voltage_branches",
                "padding",
                "dc_current",
                "dc_conductance",
                "gamma",
                "ac_current_abs",
            ]
        ]
    )
