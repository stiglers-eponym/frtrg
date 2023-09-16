# FRTRG
Floquet real-time renormalization group (RTRG) for the isotropic single-channel Kondo model.
The method implemented here can be used to describe strongly correlated open quantum systems at zero temperature with time-dependent driving.

Publications:
* [my thesis](https://doi.org/10.18154/RWTH-2023-05062)
  contains the latest and most complete documentation of the method implemented here
* [Phys. Rev. B **106**, 115440](https://doi.org/10.1103/PhysRevB.106.115440)
  ([arXiv:2206.06263](https://arxiv.org/abs/2206.06263))
  contains a more compact explanation of the method based on an earlier version of the code.
  Also have a look at the [interactive plots](https://vbruch.eu/frtrg.html).

The modules in this package are mainly intended as a reference for these publications,
and as a possible source of inspiration for other applications of Floquet or RTRG methods.


## Building and installing
To install the full package, run (the file name in the last line may vary):
```sh
git clone https://github.com/stiglers-eponym/frtrg
cd frtrg
python -m build
pip install --user dist/frtrg-0.14.16-cp311-cp311-linux_x86_64.whl
```

## Documentation
An automatically generated documentation of the python package can be found here:
<https://stiglers-eponym.github.io/frtrg/>.
**If you wish to use this module, do not hesitate to ask for better documentation!**
The python modules contain docstrings of varying quality. Only the
documentation of the kondo module is formatted according to some standards.


## Usage
This code can be used as a python library, or to generate and save data.
The individual modules contain a brief documentation in the beginning.

### Library
The FRTRG module provides the following submodules:
* `kondo`: FRTRG solution of the isotropic spin-1/2 Kondo model with two reservoirs. Create a `Kondo` object and use `Kondo.run()` to run the RG flow.
* `data_management`: save and load Kondo RG flow results in a database and in HDF5 files. Use `DataManager.list()` to obtain main results (such as the conductance) in a pandas table.
* `gen_data`: interface for generating data for AC driving from the command line and for saving the results.
* `gen_pulse_data`: interface for generating data for pulsed driving from the command line and for saving the results.
* `visualize`: basic 3d visualization command line interface using PyQtGraph
* `rtrg`: helper module for Floquet matrix RG objects
* `compact_rtrg`: variant of `rtrg` that uses (and requires) a symmetry in the bias voltage of the form V(-t)=-V(t).
* `reservoirmatrix`: helper module for matrices in reservoir space, which are also Floquet matrices and RG objects
* `rtrg_c`: helper module for modified matrix product of Floquet matrices
* `frequency_integral`: helper module for exact and approximate solutions of an integral required in the RG equations.
* `settings`: global settings

### Generating data
Results for the isotropic spin-1/2 Kondo model with two reservoirs and a bias voltage of the form `V(t) = Vdc + Vac cos(Ωt)` can be calculated from the command line as in the following example:
```sh
OMP_NUM_THREADS=1 python -m frtrg.gen_data \
    --filename=data.h5 \
    --db_filename=database.sqlite \
    --threads=1 \
    --method=mu \
    --omega=3.4425 \
    --vdc=0 \
    --vac=3.4425 \
    --voltage_branches=0 \
    --nmax=12 \
    --log_time=10
```
In this example, the results will be saved in a file `data.h5`. Metadata and main results (DC and AC current, conductance) are also saved in `database.sqlite`.
It is possible to create arrays data points with a single command line. See `python -m frtrg.gen_data --help` for more information.

### Visualization
A basic visualization using PyQtGraph is available in the `visualize` submodule (requires matplotlib and PyQtGraph):
```sh
python -m frtrg.visualize --omega=3.4425
```

### Exporting data
To select, export or visualize data, you can access a pandas table. Example:
```python
from frtrg.data_management import DataManager
dm = DataManager()
table = dm.list(omega=3.4425)
```
This assumes that the correct database is set in settings. To change this database, use
```python
from frtrg import settings
settings.defaults.DB_CONNECTION_STRING='sqlite:////path/to/database.sqlite'
settings.defaults.update_globals()
```
