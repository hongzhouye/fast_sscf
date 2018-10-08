# A Fast and Scalable Implementation of sigma-SCF

### Introduction

[sigma-SCF](https://aip.scitation.org/doi/10.1063/1.5001262) is a computational method developed recently by our research group to compute mean-field excited states. These states are traditionally computed using either linear response-based methods such as configuration interaction singles (CIS) and time-dependent density functional theory (TDDFT) or excited-state self-consistent field (SCF) methods such as Delta-SCF. Several deficiencies are known for the existing methods:
1. For linear-respose methods, orbital relaxation and double or higher excitations are missing, which make them in appropriate for many interesting applications such as charge-transfer states in multi-chromophores and double excitations in transition-metal complexes.
2. Excited-state SCF methods –– though include full orbital relaxation and can compute any degree of excitation –– tend to fall into the ground state during the orbital relaxation. This problem arises from energy being saddle point for excited states and is known as "variational collapse".

sigma-SCF solves the variational collapse of Delta-SCF by minimizing the energy variance as opposed to energy. Since energy variance is a local minimum for every state, only ground-state formalism is needed. Any desired excited state in sigma-SCF is targeted by specifying a singlet parameter –– a guess of the energy of that state. This feature makes sigma-SCF capable of finding all states within an energy window.

This project aims to develop a fast and scalable implementation for sigma-SCF so that it could be applied to molecules of ~2000 basis functions.


### Install

To test the preliminary version, change directory to `int_direct`, modify `Makefile` appropriately, and type
```
make -j2
```
to compile the source code. Make sure you have `libint` installed, which can be found [here](https://github.com/evaleev/libint). Once done, you will see a binary file named `sigma-scf.out`, which takes two command line arguments: a path to an xyz file and a string of basis name. For example
```
./sigma-scf.out ../geom/h2o.xyz cc-pVDZ
```
will output something like
```
Iter         E(HF)              var       D(var)/var   RMS([F,D])/nn   Time(s)
 01     -75.969415811       1.025264017   1.00000e+00   3.19620e-03    0.04747
 02     -75.916434197       1.097711985   6.59991e-02   2.38128e-03    0.04001
 03     -75.524749928       2.129838477   4.84603e-01   6.30636e-03    0.03713
 ...
 20     -75.947575548       0.944399778   3.16258e-10   2.86806e-08    0.03952
 21     -75.947574459       0.944399778   3.73719e-13   1.57880e-08    0.04012
 22     -75.947567690       0.944399778   3.08915e-10   7.43143e-09    0.04260
** t(Fock build)/cycle =       0.041806409091
** Hartree-Fock energy =     -75.947567690255
```
For more timing data, try
```
make clean
make -j2 extra=-DTIMING
./sigma-scf.out ../geom/h2o.xyz cc-pVDZ
```
which also outputs timing for intermediate steps,
```
Iter         E(HF)              var       D(var)/var   RMS([F,D])/nn   Time(s)
 01     -75.969415811       1.025264017   1.00000e+00   3.19620e-03    0.04306
  time-int:    0.02274
  time-mm :    0.00124
time-forming Vmnjb: 0.02505
time-forming QPQ : 0.002402
time-forming PQP : 0.000678
  time-int:    0.01905
  time-mm :    0.00101
...
```
