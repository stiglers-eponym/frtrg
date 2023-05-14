#!/usr/bin/env python

# Copyright 2022 Valentin Bruch <valentin.bruch@rwth-aachen.de>
# License: MIT
"""
Kondo FRTRG, generate interactive plot using PyQtGraph
"""

import os
import argparse
import numpy as np
from pyqtgraph.Qt import QtGui
from pyqtgraph.Qt.QtWidgets import QApplication
import pyqtgraph.opengl as gl
from matplotlib import colormaps as cm
import pandas as pd
from frtrg import settings
from frtrg.data_management import DataManager


def fixed_parameter(dm,
        vac = None,
        vdc = None,
        omega = None,
        scale = 80*np.pi,
        size = None,
        grid = False,
        xyscale = "linear",
        zscale = "linear",
        gl_preset = "translucent",
        plot_value = "dc_conductance",
        mirror_vdc = True,
        cmap = "viridis",
        show_gpat = False,
        **parameters):
    """
    Show overview of all data where one physical parameter is fixed
    """
    if plot_value == "dc_current":
        mirror_vdc = False
    if omega is not None:
        parameter = "omega"
    elif vac is not None:
        parameter = "vac"
    elif vdc is not None:
        parameter = "vdc"
    data = dm.list(vac=vac, vdc=vdc, omega=omega, **parameters)
    if mirror_vdc and parameter != "vdc" and xyscale != "log":
        data_mirror = data.copy()
        data_mirror.vdc *= -1
        data = pd.concat((data, data_mirror))
        del data_mirror
    app = QApplication([])
    w = gl.GLViewWidget()
    w.show()
    w.setWindowTitle(f"Kondo model: fixed {parameter}")
    if grid:
        gx = gl.GLGridItem()
        gy = gl.GLGridItem()
        gz = gl.GLGridItem()
        gx.translate(10, 10, 1e-6)
        gy.translate(10, 10, 1e-6)
        gz.translate(10, 10, 1e-6)
        gx.rotate(-90, 0, 1, 0)
        gy.rotate(90, 1, 0, 0)
        w.addItem(gx)
        w.addItem(gy)
        w.addItem(gz)
    ax = gl.GLAxisItem(size=QtGui.QVector3D(100,100,100))
    w.addItem(ax)
    cmap = cm.get_cmap(cmap)
    # overview
    pos = np.array([
        *(getattr(data, name) \
                for name in ("vdc", "vac", "omega") \
                if name != parameter),
        data[plot_value]
        ]).T
    xL = parameters.get("xL", 0.5)
    scale *= 1/(4*xL*(1-xL))
    if xyscale == "log":
        pos[:,0] = np.log10(pos[:,0])
        pos[:,1] = np.log10(pos[:,1])
    if zscale == "log":
        pos[:,2] = np.log10(pos[:,2])
    if size is None:
        size = scale**0.5/20
    pos[:,2] *= scale
    color_data = data[plot_value].copy()
    color_data -= color_data.min()
    color_data /= color_data.max()
    sp = gl.GLScatterPlotItem(pos=pos, size=size, color=cmap(color_data), pxMode=False)
    sp.setGLOptions(gl_preset)
    w.addItem(sp)
    app.exec()


def main(dm, **parameters):
    for name in ("vdc", "vac", "omega"):
        if parameters.get(name, None) is not None:
            fixed_parameter(dm, **parameters)
            return 0
    print("""Invalid input: Please provide one of the parameters omega, vdc, and vac.
Example: python -m frtrg.visualize --omega=3.4425""")
    return 1

def parse():
    """
    Parse command line arguments and call main()
    """
    from logging import _levelToName
    parser = argparse.ArgumentParser(description="Generate 3d overview plots using pyqtgraph")
    parser.add_argument("--db_filename", metavar="file", type=str,
            help = "SQLite database file for saving metadata")
    parser.add_argument("--log_level", metavar="str", type=str,
            default = _levelToName.get(settings.logger.level, "INFO"),
            choices = ("INFO", "DEBUG", "WARNING", "ERROR"),
            help = "logging level")
    parser.add_argument("--omega", type=float,
            help="Frequency, units of Tk")
    parser.add_argument("--method", type=str, choices=("J", "mu"),
            help="method: J or mu")
    parser.add_argument("--nmax", metavar="int", type=int,
            help="Floquet matrix size")
    parser.add_argument("--padding", metavar="int", type=int,
            help="Floquet matrix ppadding")
    parser.add_argument("--voltage_branches", metavar="int", type=int,
            help="Voltage branches")
    parser.add_argument("--resonant_dc_shift", metavar="int", type=int,
            help="resonant DC shift")
    parser.add_argument("--vdc", metavar="float", type=float,
            help="Vdc, units of Tkrg")
    fourier_coef_group = parser.add_mutually_exclusive_group()
    fourier_coef_group.add_argument("--vac", metavar="float", type=float,
            help="Vac, units of Tkrg")
    fourier_coef_group.add_argument("--fourier_coef", metavar="tuple", type=float, nargs="*",
            help="Voltage Fourier arguments, units of omega")
    parser.add_argument("--d", metavar="float", type=float,
            help="D (UV cutoff), units of Tkrg")
    parser.add_argument("--xL", metavar="float", type=float, default=0.5,
            help="Asymmetry, 0 < xL < 1")
    parser.add_argument("--truncation_order", metavar='int', type=int, choices=(2,3),
            help = "Truncation order of RG equations.")
    parser.add_argument("--compact", metavar="int", type=int,
            help="compact FRTRG implementation (0,1, or 2)")
    parser.add_argument("--solver_tol_rel", metavar="float", type=float,
            help="Solver relative tolerance")
    parser.add_argument("--solver_tol_abs", metavar="float", type=float,
            help="Solver absolute tolerance")
    parser.add_argument("--xyscale", type=str, choices=("log", "lin"), default="lin",
            help="Scale for x and y axes")
    parser.add_argument("--zscale", type=str, choices=("log", "lin"), default="lin",
            help="Scale for z axis")
    parser.add_argument("--scale", metavar="float", type=float, default=80*np.pi,
            help="Scaling factor data axis")
    parser.add_argument("--cmap", metavar="colormap", type=str, default="viridis",
            help="name of matplotlib colormap")
    parser.add_argument("--grid", metavar="bool", type=int, default=0,
            help="Show grid")
    parser.add_argument("--size", metavar="float", type=float, default=None,
            help="size of points in plot")
    parser.add_argument("--gl_preset", type=str, default="additive",
            choices=("translucent", "additive", "opaque"),
            help="OpenGL setup for drawing.")
    parser.add_argument("--integral_method", metavar="int", type=int,
            help="method for solving frequency integral (-1 for exact solution)")
    parser.add_argument("--include_Ga", metavar="bool", type=int,
            help="include Ga in RG equations")
    parser.add_argument("--min_version_major", metavar="int", type=int, default=14,
            help="Minimal major version")
    parser.add_argument("--min_version_minor", metavar="int", type=int, default=-1,
            help="Minimal minor version")
    parser.add_argument("--good_flags", metavar="int", type=int, default=0,
            help="Required solver flags")
    parser.add_argument("--bad_flags", metavar="int", type=int, default=0,
            help="Forbidden solver flags")
    args = parser.parse_args()

    options = args.__dict__
    options["min_version"] = (options.pop("min_version_major"), options.pop("min_version_minor"), -1, -1)
    db_filename = options.pop("db_filename", None)
    if db_filename is not None:
        settings.defaults.DB_CONNECTION_STRING = "sqlite:///" + os.path.abspath(db_filename)
    settings.defaults.logger.setLevel(options.pop("log_level"))
    settings.defaults.update_globals()

    dm = DataManager()
    main(dm, **options)


if __name__ == "__main__":
    parse()
