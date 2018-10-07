#ifndef _SSCF_UTILS_H_
#define _SSCF_UTILS_H_


// standard C++ headers
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cstdio>

// Eigen matrix algebra library
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// Libint Gaussian integrals library
#include <libint2/diis.h>
#include <libint2/util/intpart_iter.h>
#include <libint2/chemistry/sto3g_atomic_density.h>
#include <libint2/lcao/molden.h>
#include <libint2.hpp>


typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Matrix;  // import dense, dynamically sized Matrix type from Eigen;
                 // this is a matrix with row-major storage (http://en.wikipedia.org/wiki/Row-major_order)
                 // to meet the layout of the integrals returned by the Libint integral library


using libint2::Shell;
using libint2::Atom;
using libint2::BasisSet;
using libint2::Operator;
using libint2::BraKet;


void dumpMat(const Matrix& A, std::string fmt="% .4f",
             std::string msg=std::string());
std::vector<Atom> read_geometry(const std::string& filename);
int count_nelectron(const std::vector<Atom>& atoms);
double compute_enuc(const std::vector<Atom>& atoms);
Matrix compute_1body_ints(const BasisSet& obs, libint2::Operator t,
                          const std::vector<Atom>& atoms = std::vector<Atom>());
std::tuple<Matrix, Matrix, double> conditioning_orthogonalizer(
  const Matrix& S, double S_condition_number_threshold);
Matrix compute_2body_fock_general(const BasisSet& obs, const Matrix& D,
                                  const BasisSet& D_bs, bool D_is_shelldiagonal,
                                  double precision);
Matrix compute_soad_from_minbs(
  const std::vector<Atom>& atoms, const BasisSet& obs, const Matrix& H,
  Matrix& X, const int ndocc);
Matrix compute_2body_fock_simple(const BasisSet& obs, const Matrix& D);
Matrix compute_2body_fock(const BasisSet& obs, const Matrix& D);

// sigma-SCF-related
double compute_sscf_variance_mo(const BasisSet& obs, const Matrix& C_occ,
                                const Matrix& C_virt);
Matrix compute_2body_fock_sscf_n4mem(const BasisSet& obs, const Matrix& C_occ,
                                     const Matrix& C_virt);
Matrix compute_2body_fock_sscf_n3mem(const BasisSet& obs, const Matrix& C_occ,
                                     const Matrix& C_virt);
Matrix compute_2body_fock_sscf_n2mem(const BasisSet& obs, const Matrix& C_occ,
                                     const Matrix& C_virt);

#endif
