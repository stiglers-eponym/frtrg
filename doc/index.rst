Floquet real-time RG implementation
===================================

This is the documentation of a python implementation of the Floquet real-time
renormalization group method for the periodically driven isotropic spin 1/2
Kondo model.

The FRTRG package provides the following submodules:
 - :py:mod:`frtrg.kondo`: FRTRG solution of the isotropic spin-1/2 Kondo model with two reservoirs. Create a :py:class:`frtrg.kondo.Kondo` object and use :py:meth:`frtrg.kondo.Kondo.run` to run the RG flow.
 - :py:mod:`frtrg.data_management`: save and load Kondo RG flow results in a database and in HDF5 files. Use :py:meth:`frtrg.data_management.DataManager.list` to obtain main results (such as the conductance) in a pandas table.
 - :py:mod:`frtrg.gen_data`: interface for generating data for AC driving from the command line and for saving the results.
 - :py:mod:`frtrg.gen_pulse_data`: interface for generating data for pulsed driving from the command line and for saving the results.
 - :py:mod:`frtrg.visualize`: basic 3d visualization command line interface using PyQtGraph
 - :py:mod:`frtrg.rtrg`: helper module for Floquet matrix RG objects
 - :py:mod:`frtrg.compact_rtrg`: variant of :py:mod:`frtrg.rtrg` that uses (and requires) a symmetry in the bias voltage of the form :math:`V(-t)=-V(t)`.
 - :py:mod:`frtrg.reservoirmatrix`: helper module for matrices in reservoir space, which are also Floquet matrices and RG objects
 - :py:mod:`frtrg.rtrg_c`: helper module for modified matrix product of Floquet matrices, implemented in C
 - :py:mod:`frtrg.frequency_integral`: helper module for exact and approximate solutions of an integral required in the RG equations.
 - :py:mod:`frtrg.settings`: global settings

This documentation is incomplete.
Currently only the main module :py:mod:`frtrg.kondo` is documented in a somewhat standardized way.


Indices and tables
==================
- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`

.. toctree::
   :maxdepth: 3

   autoapi/frtrg/index


See Also
========
- package website: https://github.com/stiglers-eponym/frtrg
- explanation of the method: https://doi.org/10.18154/RWTH-2023-05062
- some interactive plots: https://vbruch.eu/frtrg.html
