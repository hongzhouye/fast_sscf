# A Fast and Scalable Implementation of sigma-SCF

### Introduction

[sigma-SCF](https://aip.scitation.org/doi/10.1063/1.5001262) is a computational method developed recently by our research group to compute mean-field excited states. These states are traditionally computed using either linear response-based methods such as configuration interaction singles (CIS) and time-dependent density functional theory (TDDFT) or excited-state self-consistent field (SCF) methods such as Delta-SCF. Several deficiencies are known for the existing methods:
1. For linear-respose methods, orbital relaxation and double or higher excitations are missing, which makes them in appropriate for many interesting applications such as charge-transfer states in multi-chromophores and double excitations in transition-metal complexes.
2. Excited-state SCF methods –– though include full orbital relaxation and can compute any degree of excitation –– tend to fall into the ground state during the orbital relaxation. This problem arises from energy being saddle point for excited states and is known as "variational collapse".

sigma-SCF solves the variational collapse of Delta-SCF by minimizing the energy variance as opposed to energy of the mean-field wave function. Since energy variance is a local minimum for every state, ground-state formalism is used all the time in sigma-SCF. A desired excited state is targeted by specifying a singlet parameter –– a guess of the energy of that state. This feature makes sigma-SCF capable of finding all states within an energy window.
