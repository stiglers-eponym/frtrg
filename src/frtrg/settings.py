# Copyright 2022 Valentin Bruch <valentin.bruch@rwth-aachen.de>
# License: MIT
"""
Kondo FRTRG, module for handling global settings

Settings for Kondo model FRTRG calculations.
This module defines default values for some settings, which can be overwritten
by environment variables. The complicated structure of these settings is not
really necessary, but I learned something from it.

You can switch between different environments:
>>> import frtrg.settings
>>> settings.env1 = settings.GlobalFlags()
>>> settings.env1.USE_REFERENCE_IMPLEMENTATION = 1
>>> settings.env2 = settings.GlobalFlags()
>>> settings.env2.IGNORE_SYMMETRIES = 1
>>> settings.env1.update_globals()
>>> # Now we are in environment 1
>>> settings.env2.update_globals()
>>> # Now we are in environment 2
>>> settings.defaults.update_globals()
>>> # Now we are in the default environment
"""

import os

try:
    import colorlog as logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(purple)s%(asctime)s%(reset)s %(log_color)s%(levelname)s%(reset)s %(message)s",
        datefmt="%H:%M:%S",
    )
except:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


class GlobalFlags:
    """
    Define global settings that should be available in all modules.

    BASEPATH
        Path to save and load files. This should be a directory.
        The database will be stored in this directory.

    FILENAME
        File name to which data should be saved, must be relative to BASEPATH

    DB_CONNECTION_STRING
        String to connect to database, e.g.:
        "sqlite:///path/to/file.sqlite"
        "mariadb+pymysql://user:password@host/dbname"

    MIN_VERSION = 12
        Minimum baseversion for loading files. Files with older version will
        be ignored.

    LOG_TIME = 10
        Log progress to stdout every LOG_TIME seconds

    ENFORCE_SYMMETRIC = 0
        Raise exception if no symmetries can be used in calculation steps.

    CHECK_SYMMETRIES = 0
        Check symmetries before each iteration of the RG equations.

    IGNORE_SYMMETRIES = 0
        Do not use any symmetries.

    EXTRAPOLATE_VOLTAGE = 0
        How to extrapolate voltage copies:
        0 means don't extrapolate but just use nearest available voltage.
        1 means do quadratic extrapolation.

    LAZY_INVERSE_FACTOR = 0.25
        Factor between 0 and 1 for truncation of extended matrix before inversion.
        0 gives most precise results, 1 means discarding padding completely in inversion.

    USE_CUBLAS = 0
        (try to) use rtrg_cublas instead of rtrg_c

    USE_REFERENCE_IMPLEMENTATION = 0
        Use the (slower) reference implementation of RG equation.
        Enabling this option also sets IGNORE_SYMMETRIES=1.
    """

    # Default values of settings. These can be overwritten directly by setting
    # environment variables.
    defaults = dict(
        BASEPATH=os.path.abspath("data"),
        DB_CONNECTION_STRING="sqlite:///"
        + os.path.join(os.path.abspath("data"), "frtrg.sqlite"),
        FILENAME="frtrg-01.h5",
        VERSION=(14, 16, -1, -1),
        MIN_VERSION=(14, 0),
        LOG_TIME=10,  # in s
        ENFORCE_SYMMETRIC=0,
        CHECK_SYMMETRIES=0,
        IGNORE_SYMMETRIES=0,
        EXTRAPOLATE_VOLTAGE=0,
        LAZY_INVERSE_FACTOR=0.25,
        USE_CUBLAS=0,
        USE_REFERENCE_IMPLEMENTATION=0,
        logger=logging.getLogger("log"),
    )

    def __init__(self):
        self.settings = {}
        self.update_globals()

    def __setattr__(self, key, value):
        if key in GlobalFlags.defaults and key != "settings":
            self.settings[key] = value
        else:
            super().__setattr__(key, value)

    def __setitem__(self, key, value):
        if key in GlobalFlags.defaults and key != "settings":
            self.settings[key] = value
        else:
            raise KeyError("invalid key: %s" % key)

    def __getattr__(self, key):
        try:
            return self.settings[key]
        except KeyError:
            try:
                return self.__class__.defaults[key]
            except KeyError:
                raise AttributeError()

    def __getitem__(self, key):
        try:
            return self.settings[key]
        except KeyError:
            return self.__class__.defaults[key]

    @classmethod
    def read_environment(cls, verbose=True):
        for key, value in cls.defaults.items():
            if key in os.environ:
                cls.defaults[key] = type(value)(os.environ[key])
                if verbose:
                    cls.defaults["logger"].info(
                        "Updated from environment: %s = %s" % (key, cls.defaults[key])
                    )
        if cls.defaults["USE_REFERENCE_IMPLEMENTATION"]:
            cls.defaults["IGNORE_SYMMETRIES"] = 1
        if "LOG_LEVEL" in os.environ:
            cls.defaults["logger"].setLevel(os.environ["LOG_LEVEL"])

    @classmethod
    def get_git_version(cls):
        try:
            process = os.popen(
                "cd %s && git rev-list --count HEAD && git rev-parse --short HEAD"
                % os.path.dirname(__file__)
            )
            git_version_strs = process.read().split("\n")
            process.close()
            cls.defaults["VERSION"] = (
                *cls.defaults["VERSION"][:2],
                int(git_version_strs[0], base=10),
                int(git_version_strs[1], base=16),
            )
        except:
            cls.defaults["logger"].warning("Getting git commit id/count version failed")

    def reset(self):
        self.settings.clear()

    def assert_compatibility(self):
        if self.USE_REFERENCE_IMPLEMENTATION:
            self.IGNORE_SYMMETRIES = 1
        assert not (self.IGNORE_SYMMETRIES and self.ENFORCE_SYMMETRIC)

    def update_globals(self):
        self.assert_compatibility()
        settings = self.__class__.defaults.copy()
        settings.update(self.settings)
        globals().update(settings)


GlobalFlags.defaults["logger"].setLevel(logging.INFO)


def export():
    return dict(
        VERSION=VERSION,
        ENFORCE_SYMMETRIC=ENFORCE_SYMMETRIC,
        CHECK_SYMMETRIES=CHECK_SYMMETRIES,
        IGNORE_SYMMETRIES=IGNORE_SYMMETRIES,
        EXTRAPOLATE_VOLTAGE=EXTRAPOLATE_VOLTAGE,
        LAZY_INVERSE_FACTOR=LAZY_INVERSE_FACTOR,
        USE_CUBLAS=USE_CUBLAS,
        USE_REFERENCE_IMPLEMENTATION=USE_REFERENCE_IMPLEMENTATION,
    )


GlobalFlags.read_environment()
GlobalFlags.get_git_version()
defaults = GlobalFlags()
