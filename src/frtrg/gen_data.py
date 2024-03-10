#!/usr/bin/env python3

# Copyright 2022 Valentin Bruch <valentin.bruch@rwth-aachen.de>
# License: MIT
"""
Kondo FRTRG, script for generating and saving data.

See help function of parser in main() for documentation.
"""

import multiprocessing as mp
import argparse
import os
import numpy as np
from logging import _levelToName
from . import settings
from .data_management import DataManager
from .kondo import Kondo


def get_ref_nmax(
    dm, omega, vdc, vac, preset="precise", dc_shift=0, ac_shift=0, nmax_shift=0
):
    parameters = dict(
        old=dict(
            omega=omega,
            solver_tol_rel=1e-8,
            solver_tol_abs=1e-10,
            d=1e9,
            method="mu",
            padding=0,
            good_flags=0x0000,
            bad_flags=0x1FFC,
            xL=0.5,
            voltage_branches=4,
        ),
        precise=dict(
            omega=omega,
            solver_tol_rel=1e-8,
            solver_tol_abs=1e-10,
            d=1e9,
            method="mu",
            padding=0,
            good_flags=0x1000,
            bad_flags=0x0FFC,
            xL=0.5,
            voltage_branches=4,
        ),
        normal=dict(
            omega=omega,
            solver_tol_rel=1e-9,
            solver_tol_abs=1e-11,
            d=1e5,
            method="mu",
            padding=0,
            good_flags=0x1000,
            bad_flags=0x0FFC,
            xL=0.5,
            voltage_branches=4,
        ),
        omega5=dict(
            omega=16.5372,
            solver_tol_rel=1e-8,
            solver_tol_abs=1e-10,
            d=1e9,
            method="mu",
            padding=0,
            good_flags=0x1000,
            bad_flags=0x0FFC,
            xL=0.5,
            voltage_branches=4,
        ),
    )
    data = dm.list(**parameters[preset])
    assert data.size > 0
    if preset == "omega5":
        vdc *= 16.5372 / omega
        vac *= 16.5372 / omega
    vdc_arr, vac_arr = np.broadcast_arrays(vdc, vac)
    nmax = -np.ones(vdc_arr.shape, dtype=np.int64).reshape((-1,))
    for i, (vdc, vac) in enumerate(
        zip((vdc_arr + dc_shift).flat, (vac_arr + ac_shift).flat)
    ):
        selection = (np.abs(data.vdc - vdc) < 1e-6) & (np.abs(data.vac - vac) < 1e-6)
        n = data.nmax[selection]
        if n.size == 1:
            nmax[i] = n.values[0]
        elif n.size > 1 and n.var() < 0.1:
            nmax[i] = round(n.mean())
    if vdc_arr.ndim == 0:
        return nmax[0] + nmax_shift
    else:
        return nmax.reshape(vdc_arr.shape) + nmax_shift


def gen_option_iter(steps=None, scale=None, **options):
    """
    Interpret given options to swipe over parameters.

    Arguments:
        steps: number of steps for each swipe dimension
        scale: spacing (linear or logarithmic) for each swipe dimension
        **options: arguments for Kondo(...) taken from an argument parser.
            These are not the options for Kondo.run(...).

    Interpretation of options is documented in the help function of this
    script (parser.epilog in main()).
    """
    iter_variables = {}
    if steps is None or len(steps) == 0:
        max_length = 1
        for key, value in options.items():
            if type(value) != list or key == "fourier_coef":
                continue
            if len(value) == 1:
                (options[key],) = value
            elif max_length == 1:
                max_length = len(value)
                iter_variables[key] = (0, value)
            else:
                assert max_length == len(value)
                iter_variables[key] = (0, value)
        for key in iter_variables.keys():
            options.pop(key)
        steps = [max_length]
    else:
        if scale is None:
            scale = len(steps) * ["linear"]
        elif isinstance(scale, str):
            scale = len(steps) * [scale]
        elif len(scale) == 1:
            scale *= len(steps)
        for key, value in options.items():
            if type(value) != list or key == "fourier_coef":
                continue
            if len(value) == 1:
                (options[key],) = value
            elif len(value) == 2:
                if scale[0] in ("lin", "linear"):
                    iter_variables[key] = (
                        0,
                        np.linspace(value[0], value[1], steps[0], dtype=type(value[0])),
                    )
                elif scale[0] in ("log", "logarithmic"):
                    iter_variables[key] = (
                        0,
                        np.logspace(
                            np.log10(value[0]),
                            np.log10(value[1]),
                            steps[0],
                            dtype=type(value[0]),
                        ),
                    )
                else:
                    raise ValueError(
                        'Unexpected value for parameters "scale": %s' % scale[0]
                    )
            elif len(value) == 3:
                dim = round(value[2])
                assert 0 <= dim < len(steps)
                if scale[dim] in ("lin", "linear"):
                    iter_variables[key] = (
                        dim,
                        np.linspace(
                            value[0], value[1], steps[dim], dtype=type(value[0])
                        ),
                    )
                elif scale[dim] == "log":
                    iter_variables[key] = (
                        dim,
                        np.logspace(
                            np.log10(value[0]),
                            np.log10(value[1]),
                            steps[dim],
                            dtype=type(value[0]),
                        ),
                    )
                else:
                    raise ValueError(
                        'Unexpected value for parameters "scale": %s' % scale[dim]
                    )
            else:
                raise ValueError(
                    "Array parameters must be of the form (start, stop, dim)"
                )
        for key in iter_variables.keys():
            options.pop(key)
    for index in np.ndindex(*steps):
        for key, (dim, array) in iter_variables.items():
            options[key] = array[index[dim]]
        result = options.copy()
        vac_omega = result.pop("vac_omega", None)
        if vac_omega is not None:
            result["vac"] = vac_omega * result["omega"]
        settings.logger.debug(
            "step %s/%s: " % (index, steps)
            + ", ".join(
                "%s=%s" % (key, value)
                for (key, value) in result.items()
                if key in iter_variables
            )
        )
        yield result


def main():
    """
    Generate and save data for FRTRG applied to the Kondo model.
    This program can generate single data points, multiple explicitly
    specified data points, or swipes over parameters.

    Energies are generally defined in units of Tkrg, the Kondo temperature
    as integration constant of the RG equations. This is related to the
    more conventional definition of the Kondo temperature by G(V=Tk)=e²/h
    (differential conductance drops to half its universal value when the DC
    bias voltage equals Tk) by Tk = 3.44249 Tkrg.
    """
    parser = argparse.ArgumentParser(
        description=main.__doc__.replace("\n    ", "\n"),
        epilog="""
There are two ways to generate multiple data points for different
parameters. All arguments which accept a list of values as input
(except fourier_coef) can be used to provide multiple values.

1.  Provide all values explicitly. Options should be given either 1 or n
    values where n is the number of data points.

Example:
    python gen_data.py --method=J --nmax 10 10 11 11 --omega=10 --vac 1 2 3 4

2.  Swipe over N different parameters p_0,...,p_{N-1} independently, where
    parameter p_i takes n_i different values in linear or logarithmic
    spacing. In this case parameter p_i gets the three arguments (minimum
    value of p_i, maximum value of p_i, and index i of the parameter):
    --p_i p_i_min p_i_max i
    If the index (i) is not given, the default value 0 is assumed.
    The numbers of values per parameter are defined by
    --steps n_0 n_1 ... n_N.
    This will iterate over n_0 × n_1 × ... × n_N parameters.
    It is possible to couple two parameters by giving both the same index.
    To use logarithmic spacing, the spacing for all dimensions must be
    provided explicitly in the form (l_i = linear or log):
    --scale l_0 l_1 ... l_N

Examples:

Swipe over vac=1,2,...,10:
    python gen_data.py --method=J --nmax=10 --omega=10 --vac 1 10 --steps=10
    or equivalently:
    python gen_data.py --method=J --nmax=10 --omega=10 --vac 1 10 0 --steps=10

Keep omega=vac and swipe over (omega,vac)=1,2,...,10 (generates 10 data points):
    python gen_data.py --method=J --nmax=10 --omega 1 10 --vac 1 10 --steps 10
    or equivalently:
    python gen_data.py --method=J --nmax=10 --omega 1 10 0 --vac 1 10 0 --steps 10

Swipe over omega=10,12,...,20 and vac=1,2,...,10 independently (generates 60 data points):
    python gen_data.py --method=J --nmax=10 --omega 10 20 0 --vac 1 10 1 --steps 6 10


Full usage example running from installed package:
OMP_NUM_THREADS=1 \\
DB_CONNECTION_STRING="sqlite:////$HOME/data/frtrg.sqlite" \\
python -m frtrg.gen_data \\
--method mu \\
--omega 10 \\
--nmax 10 20 1 \\
--voltage_branches 3 \\
--vdc 0 50 0 \\
--vac 2 16 1 \\
--steps 51 8 \\
--save reduced \\
--rtol=1e-8 \\
--atol=1e-10 \\
--d=1e9 \\
--threads=4 \\
--log_time=-1 \\
--filename $HOME/data/frtrg-01.h5
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )

    # Options for parallelization and swiping over parameters
    parallel_group = parser.add_argument_group(
        title="Parallelization, swiping over parameters"
    )
    parallel_group.add_argument(
        "--steps",
        metavar="int",
        type=int,
        nargs="+",
        help="Number of steps, provided for each independent parameter swipe dimension",
    )
    parallel_group.add_argument(
        "--scale",
        type=str,
        nargs="+",
        default="linear",
        choices=("linear", "lin", "log"),
        help="Scale used for swipes (must get same number of options as --steps)",
    )
    parallel_group.add_argument(
        "--threads",
        type=int,
        metavar="int",
        default=4,
        help="Number parallel processes (set to 0 to use all CPUs)",
    )

    # Saving
    save_group = parser.add_argument_group(title="Saving data")
    save_group.add_argument(
        "--find_pole",
        type=int,
        default=0,
        choices=(0, 1),
        help="Find and save pole of Γ",
        metavar="int",
    )
    save_group.add_argument(
        "--save",
        type=str,
        default="all",
        choices=("all", "reduced", "observables", "minimal"),
        help="select which part of the Floquet matrices should be saved",
    )
    save_group.add_argument(
        "--filename",
        metavar="file",
        type=str,
        default=os.path.join(settings.BASEPATH, settings.FILENAME),
        help="HDF5 file to which data should be saved",
    )
    save_group.add_argument(
        "--db_filename",
        metavar="file",
        type=str,
        help="SQLite database file for saving metadata",
    )

    # Physical parameters
    phys_group = parser.add_argument_group(title="Physical parameters")
    phys_group.add_argument(
        "--omega",
        metavar="float",
        type=float,
        required=True,
        nargs="+",
        help="Frequency, units of Tkrg",
    )
    phys_group.add_argument(
        "--vdc",
        metavar="float",
        type=float,
        nargs="+",
        default=0.0,
        help="Vdc, units of Tkrg",
    )
    fourier_coef_group = phys_group.add_mutually_exclusive_group()
    fourier_coef_group.add_argument(
        "--vac",
        metavar="float",
        type=float,
        nargs="+",
        default=0.0,
        help="Vac, units of Tkrg",
    )
    fourier_coef_group.add_argument(
        "--vac_omega", metavar="float", type=float, nargs="+", help="Vac/omega"
    )
    fourier_coef_group.add_argument(
        "--fourier_coef",
        metavar="tuple",
        type=float,
        nargs="*",
        help="Voltage Fourier arguments, units of Tkrg",
    )
    phys_group.add_argument(
        "--fourier_phases",
        metavar="tuple",
        type=float,
        nargs="*",
        help="Phase of voltage Fourier arguments divided by π",
    )
    phys_group.add_argument(
        "--xL",
        metavar="float",
        type=float,
        nargs="+",
        default=0.5,
        help="Asymmetry, 0 < xL < 1",
    )

    # Method parameters
    method_group = parser.add_argument_group(title="Method")
    method_group.add_argument(
        "--method",
        type=str,
        required=True,
        choices=("J", "mu"),
        help="J: include all time dependence in coupling by unitary transformation.\nmu: describe time dependence by Floquet matrix for chemical potentials.",
    )
    method_group.add_argument(
        "--simplified_initial_conditions",
        metavar="bool",
        type=int,
        default=0,
        help="Set initial condition for gammaL to 0",
    )
    method_group.add_argument(
        "--improved_initial_conditions",
        metavar="bool",
        type=int,
        default=1,
        help="Set initial condition for dgammaL/dE to linear response estimate",
    )
    method_group.add_argument(
        "--include_Ga",
        metavar="int",
        type=int,
        default=1,
        choices=(0, 1),
        help="include vertex parameter Ga in RG equations",
    )
    method_group.add_argument(
        "--solve_integral_exactly",
        metavar="bool",
        type=int,
        default=0,
        help="Solve integral in RG equations exactly by diagonalizing Floquet matrices. Requires --integral_method",
    )
    method_group.add_argument(
        "--integral_method",
        metavar="int",
        type=int,
        default=-15,
        help="Select solution/approximation of frequency integral",
    )
    method_group.add_argument(
        "--d",
        metavar="float",
        type=float,
        nargs="+",
        default=1e9,
        help="D (UV cutoff), units of Tkrg",
    )
    method_group.add_argument(
        "--resonant_dc_shift",
        metavar="int",
        type=int,
        nargs="+",
        default=0,
        help="Describe DC voltage (partially) by shift in Floquet matrices. --vdc is the full voltage.",
    )
    method_group.add_argument(
        "--truncation_order",
        metavar="int",
        type=int,
        choices=(2, 3),
        default=3,
        help="Truncation order of RG equations.",
    )
    method_group.add_argument(
        "--preset",
        metavar="str",
        type=str,
        choices=("normal", "precise", "old", "omega5"),
        default="precise",
        help="Preset for choosing nmax if set explicitly",
    )
    method_group.add_argument(
        "--preset_dc_shift",
        metavar="float",
        type=float,
        default=0,
        help="Shift in Vdc for getting nmax from existing data",
    )
    method_group.add_argument(
        "--preset_ac_shift",
        metavar="float",
        type=float,
        default=0,
        help="Shift in Vac for getting nmax from existing data",
    )
    method_group.add_argument(
        "--preset_correction",
        metavar="int",
        type=int,
        default=0,
        help="add this value to preset nmax",
    )

    # Numerical parameters concerning Floquet matrices
    numerics_group = parser.add_argument_group(title="Numerical parameters")
    numerics_group.add_argument(
        "--nmax",
        metavar="int",
        type=int,
        nargs="+",
        required=True,
        help="Floquet matrix size",
    )
    numerics_group.add_argument(
        "--padding",
        metavar="int",
        type=int,
        nargs="+",
        default=0,
        help="Floquet matrix ppadding",
    )
    numerics_group.add_argument(
        "--voltage_branches",
        metavar="int",
        type=int,
        required=True,
        help="Voltage branches",
    )
    numerics_group.add_argument(
        "--compact",
        metavar="{0,1,2}",
        type=int,
        nargs="+",
        default=0,
        help="compact FRTRG implementation (0, 1, or 2)",
    )
    numerics_group.add_argument(
        "--lazy_inverse_factor",
        metavar="float",
        type=float,
        default=settings.LAZY_INVERSE_FACTOR,
        help="Factor between 0 and 1 for truncation of extended matrix before inversion.\n0 gives most precise results, 1 means discarding padding completely in inversion.\nOverwrites value set by environment variable LAZY_INVERSE_FACTOR.",
    )
    numerics_group.add_argument(
        "--extrapolate_voltage",
        metavar="bool",
        type=int,
        default=settings.EXTRAPOLATE_VOLTAGE,
        help="Extrapolate along voltage branches (quadratic extrapolation).\nOverwrites value set by environment variable EXTRAPOLATE_VOLTAGE.",
    )
    numerics_group.add_argument(
        "--check_symmetries",
        metavar="bool",
        type=int,
        default=settings.CHECK_SYMMETRIES,
        help="Check symmetries during RG flow.\nOverwrites value set by environment variable CHECK_SYMMETRIES.",
    )
    symmetry_group = numerics_group.add_mutually_exclusive_group()
    symmetry_group.add_argument(
        "--ignore_symmetries",
        metavar="bool",
        type=int,
        default=settings.IGNORE_SYMMETRIES,
        help="Do not use any symmetries.\nOverwrites value set by environment variable IGNORE_SYMMETRIES.",
    )
    symmetry_group.add_argument(
        "--enforce_symmetric",
        metavar="bool",
        type=int,
        default=settings.ENFORCE_SYMMETRIC,
        help="Enforce using symmetries, throw errors if no symmetries can be used.\nOverwrites value set by environment variable ENFORCE_SYMMETRIC.",
    )
    numerics_group.add_argument(
        "--use_reference_implementation",
        metavar="bool",
        type=int,
        default=settings.USE_REFERENCE_IMPLEMENTATION,
        help="Use slower reference implementation of RG equations instead of optimized implementation.\nOverwrites value set by environment variable USE_REFERENCE_IMPLEMENTATION.",
    )

    # Convergence parameters concerning solver and D convergence
    solver_group = parser.add_argument_group("Solver")
    solver_group.add_argument(
        "--rtol",
        metavar="float",
        type=float,
        default=1e-8,
        help="Solver relative tolerance",
    )
    solver_group.add_argument(
        "--atol",
        metavar="float",
        type=float,
        default=1e-10,
        help="Solver relative tolerance",
    )
    solver_group.add_argument(
        "--solver_method",
        metavar="str",
        type=str,
        default="RK45",
        help="ODE solver algorithm",
    )

    # Output
    log_group = parser.add_argument_group(title="Console output")
    log_group.add_argument(
        "-h", "--help", action="help", help="show help message and exit"
    )
    log_group.add_argument(
        "--log_level",
        metavar="str",
        type=str,
        default=_levelToName.get(settings.logger.level, "INFO"),
        choices=("INFO", "DEBUG", "WARNING", "ERROR"),
        help="logging level",
    )
    log_group.add_argument(
        "--log_time",
        metavar="int",
        type=int,
        default=settings.LOG_TIME,
        help="log time interval, in s",
    )

    args = parser.parse_args()
    options = args.__dict__

    # extract options that are handled by data management and not by Kondo module
    threads = options.pop("threads")
    filename = options.pop("filename")
    include = options.pop("save")

    fourier_phases = options.pop("fourier_phases", None)
    if fourier_phases is not None:
        fourier_coef = options.get("fourier_coef")
        assert len(fourier_coef) == len(fourier_phases)
        options["fourier_coef"] = tuple(
            c * np.exp(1j * np.pi * p) for c, p in zip(fourier_coef, fourier_phases)
        )

    # update settings
    db_filename = options.pop("db_filename", None)
    if db_filename is not None:
        settings.defaults.DB_CONNECTION_STRING = "sqlite:///" + os.path.abspath(
            db_filename
        )
    for name in settings.GlobalFlags.defaults.keys():
        try:
            value = options.pop(name.lower())
            if value is not None:
                settings.defaults[name] = value
        except KeyError:
            pass
    if options["log_level"] == "DEBUG":
        settings.logging.getLogger("sqlalchemy.engine").setLevel(settings.logging.INFO)
    settings.defaults.logger.setLevel(options.pop("log_level"))
    settings.defaults.update_globals()

    find_pole = bool(options.pop("find_pole", 0))

    # Translate method argument for Kondo(...) arguments
    options.update(unitary_transformation=options.pop("method") == "J")
    # extract options for solver that are passed to Kondo.run(...) instead of Kondo(...)
    solver_options = dict(
        rtol=options.pop("rtol"),
        atol=options.pop("atol"),
        method=options.pop("solver_method"),
    )
    preset_options = dict(
        preset=options.pop("preset", "precise"),
        dc_shift=options.pop("preset_dc_shift", 0),
        ac_shift=options.pop("preset_ac_shift", 0),
        nmax_shift=options.pop("preset_correction", 0),
    )

    # Detect number of CPUs (if necessary)
    if threads == 0:
        threads = mp.cpu_count()

    # Generate data
    if threads == 1:
        # no parallelization
        dm = DataManager()
        for kondo_options in gen_option_iter(**options):
            if kondo_options["voltage_branches"] == 0:
                vdc = kondo_options["resonant_dc_shift"] * kondo_options["omega"]
                assert np.abs(kondo_options["vdc"] - vdc) < 1e-9
                kondo_options["vdc"] = vdc
            if kondo_options["nmax"] < 0:
                kondo_options["nmax"] = get_ref_nmax(
                    dm,
                    kondo_options["omega"],
                    kondo_options["vdc"],
                    kondo_options["vac"],
                    **preset_options,
                )
            if kondo_options["nmax"] < 0:
                settings.logger.error(
                    f"Invalid value nmax={nmax}: must be ≥0 at Vdc={kondo_options['vdc']:.8g}, Vac={kondo_options['vac']:.8g}, Ω={kondo_options['omega']:.8g}"
                )
                continue
            settings.logger.info(
                f"Starting with Vdc={kondo_options['vdc']:.8g}, Vac={kondo_options['vac']:.8g}, Ω={kondo_options['omega']:.8g}, nmax={kondo_options['nmax']}"
            )
            if kondo_options["padding"] < 0:
                kondo_options["padding"] = int(kondo_options["nmax"] * 0.667)
                settings.logger.info(f"...using padding={kondo_options['padding']}")
            settings.logger.debug(kondo_options)
            settings.logger.debug(solver_options)
            kondo = Kondo(**kondo_options)
            kondo.run(**solver_options)
            settings.logger.info(
                f"Saving Vdc={kondo_options['vdc']:.8g}, Vac={kondo_options['vac']:.8g}, Ω={kondo_options['omega']:.8g} to {filename}"
            )
            kid, khash = dm.save_h5(kondo, filename, include)
            if find_pole:
                pole, solver = kondo.findPole(**solver_options)
                settings.logger.info(
                    f"Saving pole for Vdc={kondo_options['vdc']:.8g}, Vac={kondo_options['vac']:.8g}, Ω={kondo_options['omega']:.8g} to database"
                )
                dm.save_pole(kid, khash, pole)
    else:
        # generate data points in parallel
        lock = mp.Lock()
        queue = mp.Queue()
        # create processes
        processes = [
            mp.Process(
                target=save_data_mp,
                args=(
                    queue,
                    lock,
                    solver_options,
                    filename,
                    include,
                    find_pole,
                    False,
                    preset_options,
                ),
            )
            for i in range(threads)
        ]
        # start processes
        for p in processes:
            p.start()
        # send data to processes
        for kondo_options in gen_option_iter(**options):
            queue.put(kondo_options)
        # send end signal to processes
        for p in processes:
            queue.put(None)


def save_data_mp(
    queue,
    lock,
    solver_options,
    filename,
    include="all",
    find_pole=False,
    overwrite=False,
    preset_options={},
):
    """
    Generate data points in own process and save them to HDF5 file.
    Each process owns one DataManager instance.
    In each run a new Kondo instance is created.
    """
    dm = DataManager()
    while True:
        kondo_options = queue.get()
        if kondo_options is None:
            break
        if kondo_options["voltage_branches"] == 0:
            vdc = kondo_options["resonant_dc_shift"] * kondo_options["omega"]
            assert np.abs(kondo_options["vdc"] - vdc) < 1e-9
            kondo_options["vdc"] = vdc
        if kondo_options["nmax"] < 0:
            kondo_options["nmax"] = get_ref_nmax(
                dm,
                kondo_options["omega"],
                kondo_options["vdc"],
                kondo_options["vac"],
                **preset_options,
            )
        if kondo_options["nmax"] < 0:
            settings.logger.error(
                f"Invalid value nmax={kondo_options['nmax']}: must be ≥0 at Vdc={kondo_options['vdc']:.8g}, Vac={kondo_options['vac']:.8g}, Ω={kondo_options['omega']:.8g}"
            )
            continue
        settings.logger.info(
            f"Starting with Vdc={kondo_options['vdc']:.8g}, Vac={kondo_options['vac']:.8g}, Ω={kondo_options['omega']:.8g}, nmax={kondo_options['nmax']}"
        )
        if kondo_options["padding"] < 0:
            kondo_options["padding"] = int(kondo_options["nmax"] * 0.667)
            settings.logger.info(f"...using padding={kondo_options['padding']}")
        settings.logger.debug(kondo_options)
        settings.logger.debug(solver_options)
        kondo = Kondo(**kondo_options)
        kondo.run(**solver_options)
        lock.acquire()
        try:
            kid = -1
            settings.logger.info(
                f"Saving Vdc={kondo_options['vdc']:.8g}, Vac={kondo_options['vac']:.8g}, Ω={kondo_options['omega']:.8g} to {filename}"
            )
            kid, khash = dm.save_h5(kondo, filename, include, overwrite)
        finally:
            lock.release()
        if find_pole and kid >= 0:
            pole, solver = kondo.findPole(**solver_options)
            settings.logger.info(
                f"Saving pole for Vdc={kondo_options['vdc']:.8g}, Vac={kondo_options['vac']:.8g}, Ω={kondo_options['omega']:.8g} to database"
            )
            dm.save_pole(kid, khash, pole)


if __name__ == "__main__":
    main()
