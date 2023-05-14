#!/usr/bin/env python3

# Copyright 2022 Valentin Bruch <valentin.bruch@rwth-aachen.de>
# License: MIT
"""
Kondo FRTRG, script for generating and saving data.

See help function of parser in main() for documentation.
"""

import argparse
import os
import numpy as np
from logging import _levelToName
from . import settings
from .data_management import DataManager
from .kondo import Kondo


def main():
    """
    Generate and save data for pulse voltage applied to the Kondo model.

    Energies are generally defined in units of Tkrg, the Kondo temperature
    as integration constant of the RG equations. This is related to the
    more conventional definition of the Kondo temperature by G(V=Tk)=e²/h
    (differential conductance drops to half its universal value when the DC
    bias voltage equals Tk) by Tk = 3.44249 Tkrg.
    """
    parser = argparse.ArgumentParser(
            description = main.__doc__.replace("\n    ", "\n"),
            formatter_class = argparse.RawDescriptionHelpFormatter,
            add_help = False)


    # Physical parameters
    phys_group = parser.add_argument_group(title="Physical parameters")
    phys_group.add_argument("pulse_shape", type=str, choices=["gauss", "gauss_symmetric", "updown_gauss"], default="gauss",
            help="name for preset parameters")
    phys_group.add_argument("--omega", metavar='float', type=float, default=0.,
            help = "Frequency, units of Tkrg")
    phys_group.add_argument("--pulse_duration", metavar='float', type=float,
            help = "duration of voltage pulses, units of 1/Tkrg")
    phys_group.add_argument("--pulse_phase", metavar='float', type=float,
            help = "phase/2π accumulated during of voltage pulses")
    phys_group.add_argument("--pulse_height", metavar='float', type=float,
            help = "height of voltage pulses")
    phys_group.add_argument("--pulse_separation", metavar='float', type=float,
            help = "separation of two voltage pulses, units of 1/Tkrg")
    phys_group.add_argument("--baseline_voltage", metavar='float', type=float, default=0.,
            help="baseline voltage, units of Tkrg")
    phys_group.add_argument("--xL", metavar='float', type=float, default=0.5,
            help = "Asymmetry, 0 < xL < 1")

    # Saving
    save_group = parser.add_argument_group(title="Saving data")
    save_group.add_argument("--save", type=str, default="all",
            choices = ("all", "reduced", "observables", "minimal"),
            help = "select which part of the Floquet matrices should be saved")
    save_group.add_argument("--filename", metavar='file', type=str,
            default = os.path.join(settings.BASEPATH, settings.FILENAME),
            help = "HDF5 file to which data should be saved")
    save_group.add_argument("--db_filename", metavar='file', type=str,
            help = "SQLite database file for saving metadata")

    # Method parameters
    method_group = parser.add_argument_group(title="Method")
    method_group.add_argument("--method", type=str, required=True, choices=('J', 'mu'),
            help = "J: include all time dependence in coupling by unitary transformation.\nmu: describe time dependence by Floquet matrix for chemical potentials.")
    method_group.add_argument("--simplified_initial_conditions", metavar="bool",
            type=int, default=0,
            help = "Set initial condition for gammaL to 0")
    method_group.add_argument("--improved_initial_conditions", metavar="bool",
            type=int, default=1,
            help = "Set initial condition for dgammaL/dE to linear response estimate")
    method_group.add_argument("--include_Ga", metavar="bool", type=int, default=0,
            help = "include vertex parameter Ga in RG equations")
    method_group.add_argument("--solve_integral_exactly", metavar="bool", type=int, default=0,
            help = "Solve integral in RG equations exactly by diagonalizing Floquet matrices. Requires --integral_method")
    method_group.add_argument("--integral_method", metavar="int", type=int, default=-15,
            help = "Select solution/approximation of frequency integral")
    method_group.add_argument("--d", metavar='float', type=float, default=1e9,
            help = "D (UV cutoff), units of Tkrg")
    method_group.add_argument("--resonant_dc_shift", metavar='int', type=int, default=0,
            help = "Describe DC voltage (partially) by shift in Floquet matrices.")
    method_group.add_argument("--truncation_order", metavar='int', type=int,
            choices=(2,3), default=3,
            help = "Truncation order of RG equations.")

    # Numerical parameters concerning Floquet matrices
    numerics_group = parser.add_argument_group(title="Numerical parameters")
    numerics_group.add_argument("--nmax", metavar='int', type=int, required=True,
            help = "Floquet matrix size")
    numerics_group.add_argument("--padding", metavar='int', type=int, default=0,
            help = "Floquet matrix ppadding")
    numerics_group.add_argument("--voltage_branches", metavar='int', type=int, required=True,
            help = "Voltage branches")
    numerics_group.add_argument("--compact", metavar='{0,1,2}', type=int, default=0,
            help = "compact FRTRG implementation (0, 1, or 2)")
    numerics_group.add_argument("--lazy_inverse_factor", metavar='float', type=float,
            default = settings.LAZY_INVERSE_FACTOR,
            help = "Factor between 0 and 1 for truncation of extended matrix before inversion.\n0 gives most precise results, 1 means discarding padding completely in inversion.\nOverwrites value set by environment variable LAZY_INVERSE_FACTOR.")
    numerics_group.add_argument("--extrapolate_voltage", metavar='bool', type=int,
            default = settings.EXTRAPOLATE_VOLTAGE,
            help = "Extrapolate along voltage branches (quadratic extrapolation).\nOverwrites value set by environment variable EXTRAPOLATE_VOLTAGE.")
    numerics_group.add_argument("--check_symmetries", metavar='bool', type=int,
            default = settings.CHECK_SYMMETRIES,
            help = "Check symmetries during RG flow.\nOverwrites value set by environment variable CHECK_SYMMETRIES.")
    symmetry_group = numerics_group.add_mutually_exclusive_group()
    symmetry_group.add_argument("--ignore_symmetries", metavar='bool', type=int,
            default = settings.IGNORE_SYMMETRIES,
            help = "Do not use any symmetries.\nOverwrites value set by environment variable IGNORE_SYMMETRIES.")
    symmetry_group.add_argument("--enforce_symmetric", metavar='bool', type=int,
            default = settings.ENFORCE_SYMMETRIC,
            help = "Enforce using symmetries, throw errors if no symmetries can be used.\nOverwrites value set by environment variable ENFORCE_SYMMETRIC.")
    numerics_group.add_argument("--use_reference_implementation", metavar='bool', type=int,
            default = settings.USE_REFERENCE_IMPLEMENTATION,
            help = "Use slower reference implementation of RG equations instead of optimized implementation.\nOverwrites value set by environment variable USE_REFERENCE_IMPLEMENTATION.")

    # Convergence parameters concerning solver and D convergence
    solver_group = parser.add_argument_group("Solver")
    solver_group.add_argument("--rtol", metavar="float", type=float, default=1e-7,
            help = "Solver relative tolerance")
    solver_group.add_argument("--atol", metavar="float", type=float, default=1e-9,
            help = "Solver relative tolerance")
    solver_group.add_argument("--solver_method", metavar="str", type=str, default="RK45",
            help = "ODE solver algorithm")

    # Output
    log_group = parser.add_argument_group(title="Console output")
    log_group.add_argument("-h", "--help", action="help",
            help = "show help message and exit")
    log_group.add_argument("--log_level", metavar="str", type=str,
            default = _levelToName.get(settings.logger.level, "INFO"),
            choices = ("INFO", "DEBUG", "WARNING", "ERROR"),
            help = "logging level")
    log_group.add_argument("--log_time", metavar="int", type=int, default=settings.LOG_TIME,
            help = "log time interval, in s")

    args = parser.parse_args()
    options = args.__dict__

    # extract options that are handled by data management and not by Kondo module
    filename = options.pop("filename")
    include = options.pop("save")

    # update settings
    db_filename = options.pop("db_filename", None)
    if db_filename is not None:
        settings.defaults.DB_CONNECTION_STRING = "sqlite:///" + os.path.abspath(db_filename)
    for name in settings.GlobalFlags.defaults.keys():
        try:
            value = options.pop(name.lower())
            if value is not None:
                settings.defaults[name] = value
        except KeyError:
            pass
    settings.defaults.logger.setLevel(options.pop("log_level"))
    settings.defaults.update_globals()

    # Translate method argument for Kondo(...) arguments
    options.update(unitary_transformation = options.pop('method') == 'J')
    # extract options for solver that are passed to Kondo.run(...) instead of Kondo(...)
    solver_options = dict(
            rtol = options.pop("rtol"),
            atol = options.pop("atol"),
            method = options.pop("solver_method"),
            )

    pulse_shape = options.pop("pulse_shape")
    if pulse_shape == "gauss":
        vdc, fourier_coef = fourier_coef_gauss(
                omega = options["omega"],
                baseline = options.pop("baseline_voltage"),
                duration = options.pop("pulse_duration"),
                height = options.pop("pulse_height"),
                phase = options.pop("pulse_phase"),
                nmax = options["nmax"],
                )
        options.pop("pulse_separation")
    elif pulse_shape == "gauss_symmetric":
        vdc, fourier_coef = fourier_coef_gauss_symmetric(
                omega = options["omega"],
                baseline = options.pop("baseline_voltage"),
                duration = options.pop("pulse_duration"),
                height = options.pop("pulse_height"),
                phase = options.pop("pulse_phase"),
                nmax = options["nmax"],
                )
        options.pop("pulse_separation")
    elif pulse_shape == "updown_gauss":
        vdc, fourier_coef = fourier_coef_updown_gauss(
                omega = options["omega"],
                baseline = options.pop("baseline_voltage"),
                duration = options.pop("pulse_duration"),
                height = options.pop("pulse_height"),
                phase = options.pop("pulse_phase"),
                separation = options.pop("pulse_separation"),
                nmax = options["nmax"],
                )
    else:
        raise ValueError(f"invalid pulse shape: {pulse_shape}")

    # Generate data
    dm = DataManager()
    if options["voltage_branches"] == 0:
        vdc_resonant = options["resonant_dc_shift"] * options["omega"]
        assert np.abs(vdc - vdc_resonant) < 1e-9
        vdc = vdc_resonant
    settings.logger.debug(options)
    settings.logger.debug(solver_options)
    kondo = Kondo(**options, vdc=vdc, fourier_coef=fourier_coef)
    kondo.run(**solver_options)
    dm.save_h5(kondo, filename, include)


def fourier_coef_gauss_symmetric(
        nmax,
        omega = None,
        duration = None,
        height = None,
        phase = None,
        baseline = 0.,
        ):
    """
    omega:  frequency, units of tkrg
    phase:  φ/2π
    duration: fwhm, units of 1/tkrg
    height: maximal pulse height, units of tkrg
    baseline: baseline voltage, units of tkrg

    returns: vdc, fourier_coef
    """
    narr = np.arange((nmax+3)//2)
    if phase is None:
        phase = height * duration / (4*(np.log(2)*np.pi)**0.5)
    elif duration is None:
        duration = phase/height * (4*(np.log(2)*np.pi)**0.5)
    elif height is None:
        height = phase/duration * (4*(np.log(2)*np.pi)**0.5)
    prefactor = height * omega * duration / (2*(np.log(2)*np.pi)**0.5)
    smoothen = (omega*duration)**2 / (4*np.log(2))
    fourier_coef = np.complex128(prefactor) * np.exp(-smoothen*narr**2)
    fourier_coef[0] += baseline
    fourier_coef_sym = np.zeros(nmax, dtype=np.complex128)
    fourier_coef_sym[::2] = fourier_coef[1:]
    return 0, fourier_coef_sym


def fourier_coef_gauss(
        nmax,
        omega,
        duration = None,
        height = None,
        phase = None,
        baseline = 0.,
        ):
    """
    omega:  frequency, units of tkrg
    phase:  φ/2π
    duration: fwhm, units of 1/tkrg
    height: maximal pulse height, units of tkrg
    baseline: baseline voltage, units of tkrg

    returns: vdc, fourier_coef
    """
    narr = np.arange(nmax+1)
    if phase is None:
        phase = height * duration / (4*(np.log(2)*np.pi)**0.5)
    elif duration is None:
        duration = phase/height * (4*(np.log(2)*np.pi)**0.5)
    elif height is None:
        height = phase/duration * (4*(np.log(2)*np.pi)**0.5)
    prefactor = height * omega * duration / (4*(np.log(2)*np.pi)**0.5)
    smoothen = (omega*duration)**2 / (16*np.log(2))
    fourier_coef = np.complex128(prefactor) * np.exp(-smoothen*narr**2)
    fourier_coef[0] += baseline
    return fourier_coef[0].real, fourier_coef[1:]


def fourier_coef_updown_gauss(
        nmax,
        omega,
        separation,
        duration = None,
        height = None,
        phase = None,
        baseline = 0.,
        ):
    assert baseline == 0.
    narr = np.arange(1, nmax+1)
    trash, coef_gauss = fourier_coef_gauss(nmax, omega, duration, height, phase, 0.)
    coef = 2j*np.sin(0.5*narr*separation*omega)*coef_gauss
    return 0., coef


if __name__ == "__main__":
    main()
