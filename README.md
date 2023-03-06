# FRTRG
Floquet real-time renormalization group (RTRG) for the isotropic single-channel Kondo model.
The method implemented here can be used to describe strongly correlated open quantum systems at zero temperature with time-dependent driving.
See [Phys. Rev. B **106**, 115440](https://doi.org/10.1103/PhysRevB.106.115440) ([arXiv:2206.06263](https://arxiv.org/abs/2206.06263)) or my [thesis](https://vbruch.eu/data/thesis.pdf) for an explanation of the method and some results.


## Building and installing
The scripts in this repository are mainly intended as a reference for publications using this code, and as a possible source of inspiration for other applications of Floquet or RTRG methods.
To install the full package run:
```sh
git clone https://github.com/stiglers-eponym/frtrg
cd frtrg
python -m build
pip install --user dist/frtrg-0.14.16-cp310-cp310-linux_x86_64.whl
```
