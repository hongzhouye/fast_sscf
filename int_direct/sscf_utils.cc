#include "sscf_utils.h"

/*
 *  Helper functions not used in main.cc
 */

// returns {X,X^{-1},rank,A_condition_number,result_A_condition_number}, where
// X is the generalized square-root-inverse such that X.transpose() * A * X = I
//
// if symmetric is true, produce "symmetric" sqrtinv: X = U . A_evals_sqrtinv .
// U.transpose()),
// else produce "canonical" sqrtinv: X = U . A_evals_sqrtinv
// where U are eigenvectors of A
// rows and cols of symmetric X are equivalent; for canonical X the rows are
// original basis (AO),
// cols are transformed basis ("orthogonal" AO)
//
// A is conditioned to max_condition_number
std::tuple<Matrix, Matrix, size_t, double, double> gensqrtinv(
    const Matrix& S, bool symmetric = false,
    double max_condition_number = 1e8) {
  Eigen::SelfAdjointEigenSolver<Matrix> eig_solver(S);
  auto U = eig_solver.eigenvectors();
  auto s = eig_solver.eigenvalues();
  auto s_max = s.maxCoeff();
  auto condition_number = std::min(
      s_max / std::max(s.minCoeff(), std::numeric_limits<double>::min()),
      1.0 / std::numeric_limits<double>::epsilon());
  auto threshold = s_max / max_condition_number;
  long n = s.rows();
  long n_cond = 0;
  for (long i = n - 1; i >= 0; --i) {
    if (s(i) >= threshold) {
      ++n_cond;
    } else
      i = 0;  // skip rest since eigenvalues are in ascending order
  }

  auto sigma = s.bottomRows(n_cond);
  auto result_condition_number = sigma.maxCoeff() / sigma.minCoeff();
  auto sigma_sqrt = sigma.array().sqrt().matrix().asDiagonal();
  auto sigma_invsqrt = sigma.array().sqrt().inverse().matrix().asDiagonal();

  // make canonical X/Xinv
  auto U_cond = U.block(0, n - n_cond, n, n_cond);
  Matrix X = U_cond * sigma_invsqrt;
  Matrix Xinv = U_cond * sigma_sqrt;
  // convert to symmetric, if needed
  if (symmetric) {
    X = X * U_cond.transpose();
    Xinv = Xinv * U_cond.transpose();
  }
  return std::make_tuple(X, Xinv, size_t(n_cond), condition_number,
                         result_condition_number);
}

// computes Superposition-Of-Atomic-Densities guess for the molecular density
// matrix
// in minimal basis; occupies subshells by smearing electrons evenly over the
// orbitals
Matrix compute_soad(const std::vector<Atom>& atoms) {
  // compute number of atomic orbitals
  size_t nao = 0;
  for (const auto& atom : atoms) {
    const auto Z = atom.atomic_number;
    nao += libint2::sto3g_num_ao(Z);
  }

  // compute the minimal basis density
  Matrix D = Matrix::Zero(nao, nao);
  size_t ao_offset = 0;  // first AO of this atom
  for (const auto& atom : atoms) {
    const auto Z = atom.atomic_number;
    const auto& occvec = libint2::sto3g_ao_occupation_vector(Z);
    for(const auto& occ: occvec) {
      D(ao_offset, ao_offset) = occ;
      ++ao_offset;
    }
  }

  return D * 0.5;  // we use densities normalized to # of electrons/2
}


/*
 *  Helper functions used in main.cc
 */

void dumpMat(const Matrix& A, std::string fmt, std::string msg)
{
  std::cout << msg << std::endl;
  const std::string fmt_dressed = fmt + std::string(" ");
  for(int i = 0; i < A.rows(); ++i)
  {
    for(int j = 0; j < A.rows(); ++j)
      printf(fmt_dressed.c_str(), A(i, j));
    std::cout << std::endl;
  }
}

std::vector<Atom> read_geometry(const std::string& filename) {
  std::cout << "Reading geometry from " << filename << std::endl;
  std::ifstream is(filename);
  if (not is.good()) {
    char errmsg[256] = "Could not open file ";
    strncpy(errmsg + 20, filename.c_str(), 235);
    errmsg[255] = '\0';
    throw std::runtime_error(errmsg);
  }

  // to prepare for MPI parallelization, we will read the entire file into a
  // string that can be
  // broadcast to everyone, then converted to an std::istringstream object that
  // can be used just like std::ifstream
  std::ostringstream oss;
  oss << is.rdbuf();
  // use ss.str() to get the entire contents of the file as an std::string
  // broadcast
  // then make an std::istringstream in each process
  std::istringstream iss(oss.str());

  // check the extension: if .xyz, assume the standard XYZ format, otherwise
  // throw an exception
  if (filename.rfind(".xyz") != std::string::npos)
    return libint2::read_dotxyz(iss);
  else
    throw "only .xyz files are accepted";
}

int count_nelectron(const std::vector<Atom>& atoms)
{
    auto nelectron = 0;
    for (auto i = 0; i < atoms.size(); ++i) nelectron += atoms[i].atomic_number;

    return nelectron;
}

double compute_enuc(const std::vector<Atom>& atoms)
{
    auto enuc = 0.0;
    for (auto i = 0; i < atoms.size(); i++)
      for (auto j = i + 1; j < atoms.size(); j++) {
        auto xij = atoms[i].x - atoms[j].x;
        auto yij = atoms[i].y - atoms[j].y;
        auto zij = atoms[i].z - atoms[j].z;
        auto r2 = xij * xij + yij * yij + zij * zij;
        auto r = sqrt(r2);
        enuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
      }

    return enuc;
}

Matrix compute_1body_ints(const BasisSet& obs, libint2::Operator obtype,
                          const std::vector<Atom>& atoms)
{
  using libint2::Shell;
  using libint2::Engine;
  using libint2::Operator;

  const auto n = obs.nbf();
  const auto nshells = obs.size();
  Matrix result(n,n);

  // construct the overlap integrals engine
  Engine engine(obtype, obs.max_nprim(), obs.max_l(), 0);
  // nuclear attraction ints engine needs to know where the charges sit ...
  // the nuclei are charges in this case; in QM/MM there will also be classical charges
  if (obtype == Operator::nuclear)
    engine.set_params(libint2::make_point_charges(atoms));

  auto shell2bf = obs.shell2bf();

  // buf[0] points to the target shell set after every call  to engine.compute()
  const auto& buf = engine.results();

  // loop over unique shell pairs, {s1,s2} such that s1 >= s2
  // this is due to the permutational symmetry of the real integrals over Hermitian operators: (1|2) = (2|1)
  for(auto s1=0; s1!=nshells; ++s1) {

    auto bf1 = shell2bf[s1]; // first basis function in this shell
    auto n1 = obs[s1].size();

    for(auto s2=0; s2<=s1; ++s2) {

      auto bf2 = shell2bf[s2];
      auto n2 = obs[s2].size();

      // compute shell pair; return is the pointer to the buffer
      engine.compute(obs[s1], obs[s2]);

      // "map" buffer to a const Eigen Matrix, and copy it to the corresponding blocks of the result
      Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
      result.block(bf1, bf2, n1, n2) = buf_mat;
      if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1} block, note the transpose!
      result.block(bf2, bf1, n2, n1) = buf_mat.transpose();

    }
  }

  return result;
}

std::tuple<Matrix, Matrix, double> conditioning_orthogonalizer(
    const Matrix& S, double S_condition_number_threshold) {
  size_t obs_rank;
  double S_condition_number;
  double XtX_condition_number;
  Matrix X, Xinv;

  assert(S.rows() == S.cols());

  std::tie(X, Xinv, obs_rank, S_condition_number, XtX_condition_number) =
      gensqrtinv(S, false, S_condition_number_threshold);
  auto obs_nbf_omitted = (long)S.rows() - (long)obs_rank;
  std::cout << "overlap condition number = " << S_condition_number;
  if (obs_nbf_omitted > 0)
    std::cout << " (dropped " << obs_nbf_omitted << " "
              << (obs_nbf_omitted > 1 ? "fns" : "fn") << " to reduce to "
              << XtX_condition_number << ")";
  std::cout << std::endl;

  if (obs_nbf_omitted > 0) {
    Matrix should_be_I = X.transpose() * S * X;
    Matrix I = Matrix::Identity(should_be_I.rows(), should_be_I.cols());
    std::cout << "||X^t * S * X - I||_2 = " << (should_be_I - I).norm()
              << " (should be 0)" << std::endl;
  }

  return std::make_tuple(X, Xinv, XtX_condition_number);
}

Matrix compute_2body_fock_general(const BasisSet& obs, const Matrix& D,
                                  const BasisSet& D_bs, bool D_is_shelldiagonal,
                                  double precision) {
  const auto n = obs.nbf();
  const auto nshells = obs.size();
  const auto n_D = D_bs.nbf();
  assert(D.cols() == D.rows() && D.cols() == n_D);

  // using libint2::nthreads;
  Matrix G = Matrix::Zero(n, n);

  // construct the 2-electron repulsion integrals engine
  using libint2::Engine;
  Engine engine(libint2::Operator::coulomb,
                std::max(obs.max_nprim(), D_bs.max_nprim()),
                std::max(obs.max_l(), D_bs.max_l()), 0);
  engine.set_precision(precision);  // shellset-dependent precision control
                                    // will likely break positive
                                    // definiteness
                                    // stick with this simple recipe
  auto shell2bf = obs.shell2bf();
  auto shell2bf_D = D_bs.shell2bf();

  const auto& buf = engine.results();

  // loop over permutationally-unique set of shells
  for (auto s1 = 0l, s1234 = 0l; s1 != nshells; ++s1) {
    auto bf1_first = shell2bf[s1];  // first basis function in this shell
    auto n1 = obs[s1].size();       // number of basis functions in this shell

    for (auto s2 = 0; s2 <= s1; ++s2) {
      auto bf2_first = shell2bf[s2];
      auto n2 = obs[s2].size();

      for (auto s3 = 0; s3 < D_bs.size(); ++s3) {
        auto bf3_first = shell2bf_D[s3];
        auto n3 = D_bs[s3].size();

        auto s4_begin = D_is_shelldiagonal ? s3 : 0;
        auto s4_fence = D_is_shelldiagonal ? s3 + 1 : D_bs.size();

        for (auto s4 = s4_begin; s4 != s4_fence; ++s4, ++s1234) {
          // uncomment this statement if parallel
          // if (s1234 % nthreads != thread_id) continue;

          auto bf4_first = shell2bf_D[s4];
          auto n4 = D_bs[s4].size();

          // compute the permutational degeneracy (i.e. # of equivalents) of
          // the given shell set
          auto s12_deg = (s1 == s2) ? 1.0 : 2.0;

          if (s3 >= s4) {
            auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
            auto s1234_deg = s12_deg * s34_deg;
            // auto s1234_deg = s12_deg;
            engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                obs[s1], obs[s2], D_bs[s3], D_bs[s4]);
            const auto* buf_1234 = buf[0];
            if (buf_1234 != nullptr) {
              for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                const auto bf1 = f1 + bf1_first;
                for (auto f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for (auto f3 = 0; f3 != n3; ++f3) {
                    const auto bf3 = f3 + bf3_first;
                    for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
                      const auto bf4 = f4 + bf4_first;

                      const auto value = buf_1234[f1234];
                      const auto value_scal_by_deg = value * s1234_deg;
                      G(bf1, bf2) += 2.0 * D(bf3, bf4) * value_scal_by_deg;
                    }
                  }
                }
              }
            }
          }

          engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
              obs[s1], D_bs[s3], obs[s2], D_bs[s4]);
          const auto* buf_1324 = buf[0];
          if (buf_1324 == nullptr)
            continue; // if all integrals screened out, skip to next quartet

          for (auto f1 = 0, f1324 = 0; f1 != n1; ++f1) {
            const auto bf1 = f1 + bf1_first;
            for (auto f3 = 0; f3 != n3; ++f3) {
              const auto bf3 = f3 + bf3_first;
              for (auto f2 = 0; f2 != n2; ++f2) {
                const auto bf2 = f2 + bf2_first;
                for (auto f4 = 0; f4 != n4; ++f4, ++f1324) {
                  const auto bf4 = f4 + bf4_first;

                  const auto value = buf_1324[f1324];
                  const auto value_scal_by_deg = value * s12_deg;
                  G(bf1, bf2) -= D(bf3, bf4) * value_scal_by_deg;
                }
              }
            }
          }
        }
      }
    }
  }

  // symmetrize the result and return
  return 0.5 * (G + G.transpose());
}

Matrix compute_soad_from_minbs(
  const std::vector<Atom>& atoms, const BasisSet& obs, const Matrix& H,
  Matrix& X, const int ndocc)
{  // use SOAD as the guess density
  const auto tstart = std::chrono::high_resolution_clock::now();

  auto D_minbs = compute_soad(atoms);  // compute guess in minimal basis
  BasisSet minbs("STO-3G", atoms);
  // if (minbs == obs)
  if (false)
    // return std::makeD_minbs;
    return Matrix();
  else {  // if basis != minimal basis, map non-representable SOAD guess
          // into the AO basis
          // by diagonalizing a Fock matrix
    std::cout << "projecting SOAD into AO basis ... ";
    Matrix F = H + compute_2body_fock_general(
        obs, D_minbs, minbs, true /* SOAD_D_is_shelldiagonal */,
        std::numeric_limits<double>::epsilon()  // this is cheap, no reason to be cheaper
        );

    // solve F C = e S C by (conditioned) transformation to F' C' = e C',
    // where
    // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
    Eigen::SelfAdjointEigenSolver<Matrix> eig_solver(X.transpose() * F * X);
    Matrix C = X * eig_solver.eigenvectors();

    const auto tstop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_elapsed = tstop - tstart;
    std::cout << "done (" << time_elapsed.count() << " s)" << std::endl;

    return C;

    // compute density, D = C(occ) . C(occ)T
    // Matrix D = C.leftCols(ndocc) * C.leftCols(ndocc).transpose();
    // Matrix Q = C * C.transpose() - D;

    // return std::make_tuple(D, Q);
  }
}

Matrix compute_2body_fock_simple(const BasisSet& obs, const Matrix& D) {
  using libint2::Shell;
  using libint2::Engine;
  using libint2::Operator;

  const auto n = obs.nbf();
  const auto nshells = obs.size();
  Matrix G = Matrix::Zero(n,n);

  // construct the electron repulsion integrals engine
  Engine engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);
  // cout << "engine done!" << endl;

  auto shell2bf = obs.shell2bf();
  // cout << "shell2bf done!" << endl;

  // buf[0] points to the target shell set after every call  to engine.compute()
  const auto& buf = engine.results();
  // cout << "bf done!" << endl;

  // loop over shell pairs of the Fock matrix, {s1,s2}
  // Fock matrix is symmetric, but skipping it here for simplicity (see compute_2body_fock)
  for(auto s1=0; s1!=nshells; ++s1) {

    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = obs[s1].size();

    for(auto s2=0; s2!=nshells; ++s2) {

      auto bf2_first = shell2bf[s2];
      auto n2 = obs[s2].size();

      // loop over shell pairs of the density matrix, {s3,s4}
      // again symmetry is not used for simplicity
      for(auto s3=0; s3!=nshells; ++s3) {

        auto bf3_first = shell2bf[s3];
        auto n3 = obs[s3].size();

        for(auto s4=0; s4!=nshells; ++s4) {

          auto bf4_first = shell2bf[s4];
          auto n4 = obs[s4].size();

          // Coulomb contribution to the Fock matrix is from {s1,s2,s3,s4} integrals
          // cout << "  compute done for " << s1 << " " << s2 << " " << s3 <<
          //   " " << s4 << endl;
          engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
          // cout << "  Done!" << endl;
          const auto* buf_1234 = buf[0];
          if (buf_1234 == nullptr)
            continue; // if all integrals screened out, skip to next quartet
          // cout << "    pass screeening" << endl;

          // we don't have an analog of Eigen for tensors (yet ... see github.com/BTAS/BTAS, under development)
          // hence some manual labor here:
          // 1) loop over every integral in the shell set (= nested loops over basis functions in each shell)
          // and 2) add contribution from each integral
          for(auto f1=0, f1234=0; f1!=n1; ++f1) {
            const auto bf1 = f1 + bf1_first;
            for(auto f2=0; f2!=n2; ++f2) {
              const auto bf2 = f2 + bf2_first;
              for(auto f3=0; f3!=n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for(auto f4=0; f4!=n4; ++f4, ++f1234) {
                  const auto bf4 = f4 + bf4_first;
                  G(bf1,bf2) += D(bf3,bf4) * 2.0 * buf_1234[f1234];
                }
              }
            }
          }
          // cout << "    pass J" << endl;

          // exchange contribution to the Fock matrix is from {s1,s3,s2,s4} integrals
          engine.compute(obs[s1], obs[s3], obs[s2], obs[s4]);
          const auto* buf_1324 = buf[0];

          for(auto f1=0, f1324=0; f1!=n1; ++f1) {
            const auto bf1 = f1 + bf1_first;
            for(auto f3=0; f3!=n3; ++f3) {
              const auto bf3 = f3 + bf3_first;
              for(auto f2=0; f2!=n2; ++f2) {
                const auto bf2 = f2 + bf2_first;
                for(auto f4=0; f4!=n4; ++f4, ++f1324) {
                  const auto bf4 = f4 + bf4_first;
                  // cout << "      " << bf1 << " " << bf2 << " " << bf3 << " "
                  //   << bf4 << " " << f1324 << endl;
                  G(bf1,bf2) -= D(bf3,bf4) * buf_1324[f1324];
                }
              }
            }
          }
          // cout << "    pass K" << endl;

        }
      }
    }
  }

  return G;
}

Matrix compute_2body_fock(const BasisSet& obs, const Matrix& D) {

  using libint2::Shell;
  using libint2::Engine;
  using libint2::Operator;

  std::chrono::duration<double> time_elapsed = std::chrono::duration<double>::zero();

  const auto n = obs.nbf();
  const auto nshells = obs.size();
  Matrix G = Matrix::Zero(n,n);

  // construct the 2-electron repulsion integrals engine
  Engine engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);

  auto shell2bf = obs.shell2bf();

  const auto& buf = engine.results();

  // The problem with the simple Fock builder is that permutational symmetries of the Fock,
  // density, and two-electron integrals are not taken into account to reduce the cost.
  // To make the simple Fock builder efficient we must rearrange our computation.
  // The most expensive step in Fock matrix construction is the evaluation of 2-e integrals;
  // hence we must minimize the number of computed integrals by taking advantage of their permutational
  // symmetry. Due to the multiplicative and Hermitian nature of the Coulomb kernel (and realness
  // of the Gaussians) the permutational symmetry of the 2-e ints is given by the following relations:
  //
  // (12|34) = (21|34) = (12|43) = (21|43) = (34|12) = (43|12) = (34|21) = (43|21)
  //
  // (here we use chemists' notation for the integrals, i.e in (ab|cd) a and b correspond to
  // electron 1, and c and d -- to electron 2).
  //
  // It is easy to verify that the following set of nested loops produces a permutationally-unique
  // set of integrals:
  // foreach a = 0 .. n-1
  //   foreach b = 0 .. a
  //     foreach c = 0 .. a
  //       foreach d = 0 .. (a == c ? b : c)
  //         compute (ab|cd)
  //
  // The only complication is that we must compute integrals over shells. But it's not that complicated ...
  //
  // The real trick is figuring out to which matrix elements of the Fock matrix each permutationally-unique
  // (ab|cd) contributes. STOP READING and try to figure it out yourself. (to check your answer see below)

  // loop over permutationally-unique set of shells
  for(auto s1=0; s1!=nshells; ++s1) {

    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = obs[s1].size();   // number of basis functions in this shell

    for(auto s2=0; s2<=s1; ++s2) {

      auto bf2_first = shell2bf[s2];
      auto n2 = obs[s2].size();

      for(auto s3=0; s3<=s1; ++s3) {

        auto bf3_first = shell2bf[s3];
        auto n3 = obs[s3].size();

        const auto s4_max = (s1 == s3) ? s2 : s3;
        for(auto s4=0; s4<=s4_max; ++s4) {

          auto bf4_first = shell2bf[s4];
          auto n4 = obs[s4].size();

          // compute the permutational degeneracy (i.e. # of equivalents) of the given shell set
          auto s12_deg = (s1 == s2) ? 1.0 : 2.0;
          auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
          auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;
          auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

          // const auto tstart = std::chrono::high_resolution_clock::now();

          engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
          const auto* buf_1234 = buf[0];
          if (buf_1234 == nullptr)
            continue; // if all integrals screened out, skip to next quartet

          // const auto tstop = std::chrono::high_resolution_clock::now();
          // time_elapsed += tstop - tstart;

          // ANSWER
          // 1) each shell set of integrals contributes up to 6 shell sets of the Fock matrix:
          //    F(a,b) += (ab|cd) * D(c,d)
          //    F(c,d) += (ab|cd) * D(a,b)
          //    F(b,d) -= 1/4 * (ab|cd) * D(a,c)
          //    F(b,c) -= 1/4 * (ab|cd) * D(a,d)
          //    F(a,c) -= 1/4 * (ab|cd) * D(b,d)
          //    F(a,d) -= 1/4 * (ab|cd) * D(b,c)
          // 2) each permutationally-unique integral (shell set) must be scaled by its degeneracy,
          //    i.e. the number of the integrals/sets equivalent to it
          // 3) the end result must be symmetrized
          for(auto f1=0, f1234=0; f1!=n1; ++f1) {
            const auto bf1 = f1 + bf1_first;
            for(auto f2=0; f2!=n2; ++f2) {
              const auto bf2 = f2 + bf2_first;
              for(auto f3=0; f3!=n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for(auto f4=0; f4!=n4; ++f4, ++f1234) {
                  const auto bf4 = f4 + bf4_first;

                  const auto value = buf_1234[f1234];

                  const auto value_scal_by_deg = value * s1234_deg;

                  G(bf1,bf2) += D(bf3,bf4) * value_scal_by_deg;
                  G(bf3,bf4) += D(bf1,bf2) * value_scal_by_deg;
                  G(bf1,bf3) -= 0.25 * D(bf2,bf4) * value_scal_by_deg;
                  G(bf2,bf4) -= 0.25 * D(bf1,bf3) * value_scal_by_deg;
                  G(bf1,bf4) -= 0.25 * D(bf2,bf3) * value_scal_by_deg;
                  G(bf2,bf3) -= 0.25 * D(bf1,bf4) * value_scal_by_deg;
                }
              }
            }
          }

        }
      }
    }
  }

  // symmetrize the result and return
  Matrix Gt = G.transpose();
  return 0.5 * (G + Gt);
}

/*
 *  Helper functions for sigma-SCF
 */

#define IND4(i,j,k,l,n) i*n*n*n+j*n*n+k*n+l

// Compute full ERI tensor (for debug!)
std::vector<double> compute_ERI(const BasisSet& obs)
{
  using libint2::Engine;
  using libint2::Operator;

  const auto n = obs.nbf();
  const auto nshells = obs.size();

  // construct the electron repulsion integrals engine
  Engine engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);

  auto shell2bf = obs.shell2bf();

  // buf[0] points to the target shell set after every call  to engine.compute()
  const auto& buf = engine.results();

  std::vector<double> ERI(n*n*n*n, 0.);
  for(auto s1=0; s1!=nshells; ++s1) {

    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = obs[s1].size();

    for(auto s2=0; s2!=nshells; ++s2) {

      auto bf2_first = shell2bf[s2];
      auto n2 = obs[s2].size();

      // loop over shell pairs of the density matrix, {s3,s4}
      // again symmetry is not used for simplicity
      for(auto s3=0; s3!=nshells; ++s3) {

        auto bf3_first = shell2bf[s3];
        auto n3 = obs[s3].size();

        for(auto s4=0; s4!=nshells; ++s4) {

          auto bf4_first = shell2bf[s4];
          auto n4 = obs[s4].size();

          engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
          const auto* buf_1234 = buf[0];
          if (buf_1234 == nullptr)
            continue; // if all integrals screened out, skip to next quartet

          for(auto f1=0, f1234=0; f1!=n1; ++f1) {
            const auto bf1 = f1 + bf1_first;
            for(auto f2=0; f2!=n2; ++f2) {
              const auto bf2 = f2 + bf2_first;
              for(auto f3=0; f3!=n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for(auto f4=0; f4!=n4; ++f4, ++f1234) {
                  const auto bf4 = f4 + bf4_first;
                  ERI[IND4(bf1,bf2,bf3,bf4,n)] = buf_1234[f1234];
                }
              }
            }
          }

        }
      }
    }
  }

  return ERI;
}

// Vmnjb(mu,nu,j,b) = (mu,nu|j,b)
// This tensor is required in all O(N^3)-memory impementation of sigma-SCF Fock
// builder. Integral screening is essential to achieve the O(N^3) storage
// scaling, but has not yet been implemented.
void compute_Vmnjb_prmt(const BasisSet& obs, const Matrix& C_occ,
                        const Matrix& C_virt, std::vector<double>& Vmnjb)
{
  using libint2::Engine;
  using libint2::Operator;

  const auto n = obs.nbf();
  const auto nocc = C_occ.cols();
  const auto nvirt = C_virt.cols();
  const auto nshells = obs.size();

  // construct the electron repulsion integrals engine
  Engine engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);

  auto shell2bf = obs.shell2bf();

  // buf[0] points to the target shell set after every call  to engine.compute()
  const auto& buf = engine.results();

#ifdef TIMING
  auto tstart = std::chrono::high_resolution_clock::now();
  auto tend = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_int = std::chrono::duration_cast<
    std::chrono::duration<double>>(tend-tstart);
  std::chrono::duration<double> time_mm = std::chrono::duration_cast<
    std::chrono::duration<double>>(tend-tstart);
#endif
  // s1~s4 are indices for the first ERI
  for(auto s1=0; s1!=nshells; ++s1) {

    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = obs[s1].size();

    for(auto s2=0; s2<=s1; ++s2) {

      auto bf2_first = shell2bf[s2];
      auto n2 = obs[s2].size();

      std::vector<double> tmp4(n1*n2*n*n, 0.);

#ifdef TIMING
      tstart = std::chrono::high_resolution_clock::now();
#endif
      // loop over shell pairs of the density matrix, {s3,s4}
      // again symmetry is not used for simplicity
      for(auto s3=0; s3!=nshells; ++s3) {

        auto bf3_first = shell2bf[s3];
        auto n3 = obs[s3].size();

        for(auto s4=0; s4<=s3; ++s4) {

          auto bf4_first = shell2bf[s4];
          auto n4 = obs[s4].size();

          engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
          const auto* buf_1234 = buf[0];
          if (buf_1234 == nullptr)
            continue; // if all integrals screened out, skip to next quartet

          for(auto f1=0, f1234=0; f1!=n1; ++f1) {
            for(auto f2=0; f2!=n2; ++f2) {
              for(auto f3=0; f3!=n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for(auto f4=0; f4!=n4; ++f4, ++f1234) {
                  const auto bf4 = f4 + bf4_first;
                  tmp4[f1*n2*n*n+f2*n*n+bf3*n+bf4] =
                    tmp4[f1*n2*n*n+f2*n*n+bf4*n+bf3] = buf_1234[f1234];
                }
              }
            }
          }

        }
      }
#ifdef TIMING
      tend = std::chrono::high_resolution_clock::now();
      time_int += std::chrono::duration_cast<
        std::chrono::duration<double>>(tend-tstart);
#endif

#ifdef TIMING
      tstart = std::chrono::high_resolution_clock::now();
#endif
      Matrix tmp2(n, n);
      for(auto f1=0; f1!=n1; ++f1) {
        const auto bf1 = f1 + bf1_first;
        for(auto f2=0; f2!=n2; ++f2) {
        // for(auto f2=0; f2<=f1; ++f2) {
          const auto bf2 = f2 + bf2_first;
          for(auto f3=0; f3!=n; ++f3)
            for(auto f4=0; f4!=n; ++f4)
              tmp2(f3, f4) = tmp4[f1*n2*n*n+f2*n*n+f3*n+f4];

          Matrix tmp22 = C_occ.transpose() * tmp2 * C_virt;

          for(auto j=0; j!=nocc; ++j)
            for(auto b=0; b!=nvirt; ++b)
              Vmnjb[bf1*n*nocc*nvirt+bf2*nocc*nvirt+j*nvirt+b] =
                Vmnjb[bf2*n*nocc*nvirt+bf1*nocc*nvirt+j*nvirt+b] = tmp22(j, b);
        }
      }
#ifdef TIMING
      tend = std::chrono::high_resolution_clock::now();
      time_mm += std::chrono::duration_cast<
        std::chrono::duration<double>>(tend-tstart);
#endif

    }
  }
#ifdef TIMING
  printf("  time-int: %10.5lf\n", time_int.count());
  printf("  time-mm : %10.5lf\n", time_mm.count());
#endif
}

// A O(N^5) time, O(N^4) memory implementation of sigma-SCF variance builder
// The O(N^4) storage comes from intermediate tensor Vmajb
// NOTE: this can be trivially improved to O(N^2) storage once we have a fast
//       implementation of the sigma-SCF Fock builder due to their formal
//       resemblence!
double compute_sscf_variance_mo(const BasisSet& obs, const Matrix& C_occ,
                                const Matrix& C_virt)
{
  const auto n = obs.nbf();
  const auto nocc = C_occ.cols();
  const auto nvirt = C_virt.cols();

  // alloc space for intermediate tensors
  // auto tstart = std::chrono::high_resolution_clock::now();
  std::vector<double> Viajb(nocc*nvirt*nocc*nvirt, 0.);
  std::vector<double> Vmnjb(n*n*nocc*nvirt, 0.);
  // auto tend = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> time_elapsed = tend - tstart;
  // std::cout << "time-initialization: " << time_elapsed.count() << std::endl;

  // integral-explicit version (for debug)
  // std::vector<double> ERI = compute_ERI(obs);
  // for(auto mu=0; mu!=n; ++mu) for(auto nu=0; nu<=mu; ++nu)
  // {
  //   Matrix tmp2(n, n);
  //   for(auto la=0; la!=n; ++la) for(auto si=0; si<=la; ++si)
  //     tmp2(la, si) = tmp2(si, la) = ERI[IND4(mu,nu,la,si,n)];
  //   Matrix tmp22 = C_occ.transpose() * tmp2 * C_virt;
  //   for(auto j=0; j!=nocc; ++j) for(auto b=0; b!=nvirt; ++b)
  //     Vmnjb[mu*n*nocc*nvirt+nu*nocc*nvirt+j*nvirt+b] =
  //       Vmnjb[nu*n*nocc*nvirt+mu*nocc*nvirt+j*nvirt+b] = tmp22(j, b);
  // }
  //
  // for(auto j=0; j!=nocc; ++j) for(auto b=0; b!=nvirt; ++b)
  // {
  //   Matrix tmp2(n, n);
  //   for(auto mu=0; mu!=n; ++mu) for(auto nu=0; nu<=mu; ++nu)
  //     tmp2(mu, nu) = tmp2(nu, mu) =
  //       Vmnjb[mu*n*nocc*nvirt+nu*nocc*nvirt+j*nvirt+b];
  //   Matrix tmp22 = C_occ.transpose() * tmp2 * C_virt;
  //   for(auto i=0; i!=nocc; ++i) for(auto a=0; a!=nvirt; ++a)
  //     Viajb[i*nvirt*nocc*nvirt+a*nocc*nvirt+j*nvirt+b] = tmp22(i, a);
  // }
  // delete Vmnjb;

  // tstart = std::chrono::high_resolution_clock::now();
  compute_Vmnjb_prmt(obs, C_occ, C_virt, Vmnjb);
  // tend = std::chrono::high_resolution_clock::now();
  // time_elapsed = tend - tstart;
  // std::cout << "time-forming Vmnjb: " << time_elapsed.count() << std::endl;

  // tstart = std::chrono::high_resolution_clock::now();
  for(auto j=0; j!=nocc; ++j)
    for(auto b=0; b!=nvirt; ++b) {
      Matrix tmp2(n, n);

      for(auto bf1=0; bf1!=n; ++bf1)
        for(auto bf2=0; bf2<=bf1; ++bf2)
          tmp2(bf1, bf2) = tmp2(bf2, bf1) =
            Vmnjb[bf1*n*nocc*nvirt+bf2*nocc*nvirt+j*nvirt+b];

      Matrix tmp22 = C_occ.transpose() * tmp2 * C_virt;

      for(auto i=0; i!=nocc; ++i)
        for(auto a=0; a!=nvirt; ++a)
          Viajb[i*nvirt*nocc*nvirt+a*nocc*nvirt+j*nvirt+b] = tmp22(i, a);
    }
  // tend = std::chrono::high_resolution_clock::now();
  // time_elapsed = tend - tstart;
  // std::cout << "time-forming Viajb: " << time_elapsed.count() << std::endl;

  // compute var from Viajb
  double var_c = 0.;
  double var_e = 0.;

  // tstart = std::chrono::high_resolution_clock::now();
  for(auto i=0; i<nvirt*nocc*nvirt*nocc; ++i)
    var_c += Viajb[i]*Viajb[i];
  for(auto i=0; i<nocc; ++i) for(auto a=0; a<nvirt; ++a)
  for(auto j=0; j<nocc; ++j) for(auto b=0; b<nvirt; ++b)
    var_e += Viajb[i*nvirt*nocc*nvirt+a*nocc*nvirt+j*nvirt+b] *
      Viajb[i*nvirt*nocc*nvirt+b*nocc*nvirt+j*nvirt+a];

  // tend = std::chrono::high_resolution_clock::now();
  // time_elapsed = tend - tstart;
  // std::cout << "time-forming vars : " << time_elapsed.count() << std::endl;

  // printf("var_c = % .10f\nvar_e = % .10f\n", var_c, var_e);
  return 2.*var_c - var_e;
}

// A O(N^5) time, O(N^4) memory implementation of sigma-SCF Fock builder
// The O(N^4) storage comes from intermediate tensor Vmajb
Matrix compute_2body_fock_sscf_n4mem(const BasisSet& obs, const Matrix& C_occ,
                                     const Matrix& C_virt)
{
  const auto n = obs.nbf();
  const auto nocc = C_occ.cols();
  const auto nvirt = C_virt.cols();

#ifdef TIMING
  auto tstart = std::chrono::high_resolution_clock::now();
#endif
  std::vector<double> Vmnjb(n*n*nocc*nvirt, 0.);
  compute_Vmnjb_prmt(obs, C_occ, C_virt, Vmnjb);
#ifdef TIMING
  auto tend = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_elapsed = tend - tstart;
  std::cout << "time-forming Vmnjb: " << time_elapsed.count() << std::endl;
#endif

#ifdef TIMING
  tstart = std::chrono::high_resolution_clock::now();
#endif
  std::vector<double> Vmajb(n*nvirt*nocc*nvirt, 0.);
  for(auto bf1=0; bf1!=n; ++bf1)
    for(auto a=0; a!=nvirt; ++a)
      for(auto j=0; j!=nocc; ++j)
        for(auto b=0; b!=nvirt; ++b)
          for(auto bf2=0; bf2!=n; ++bf2)
            Vmajb[bf1*nvirt*nocc*nvirt+a*nocc*nvirt+j*nvirt+b] +=
              C_virt(bf2, a) * Vmnjb[bf1*n*nocc*nvirt+bf2*nocc*nvirt+j*nvirt+b];
#ifdef TIMING
  tend = std::chrono::high_resolution_clock::now();
  time_elapsed = tend - tstart;
  std::cout << "time-forming Vmajb: " << time_elapsed.count() << std::endl;
#endif

#ifdef TIMING
  tstart = std::chrono::high_resolution_clock::now();
#endif
  std::vector<double> Vinjb(nocc*n*nocc*nvirt, 0.);
  for(auto bf2=0; bf2!=n; ++bf2)
    for(auto i=0; i!=nocc; ++i)
      for(auto j=0; j!=nocc; ++j)
        for(auto b=0; b!=nvirt; ++b)
          for(auto bf1=0; bf1!=n; ++bf1)
            Vinjb[bf2*nocc*nocc*nvirt+i*nocc*nvirt+j*nvirt+b] +=
              C_occ(bf1, i) * Vmnjb[bf1*n*nocc*nvirt+bf2*nocc*nvirt+j*nvirt+b];
#ifdef TIMING
  tend = std::chrono::high_resolution_clock::now();
  time_elapsed = tend - tstart;
  std::cout << "time-forming Vinjb: " << time_elapsed.count() << std::endl;
#endif

#ifdef TIMING
  tstart = std::chrono::high_resolution_clock::now();
#endif
  Matrix Fsscf_c = Matrix::Zero(n, n);
  Matrix Fsscf_e = Matrix::Zero(n, n);
  const auto nijb = nocc*nocc*nvirt;
  const auto najb = nvirt*nocc*nvirt;
  for(auto bf1=0; bf1!=n; ++bf1)
    for(auto bf2=0; bf2<=bf1; ++bf2)
    {
      // Coulomb-like
      // QPQ
      for(auto i=0; i!=najb; ++i)
        Fsscf_c(bf1, bf2) += Vmajb[bf1*najb+i] * Vmajb[bf2*najb+i];
      // PQP
      for(auto i=0; i!=nijb; ++i)
        Fsscf_c(bf1, bf2) -= Vinjb[bf1*nijb+i] * Vinjb[bf2*nijb+i];
      Fsscf_c(bf2, bf1) = Fsscf_c(bf1, bf2);

      // Exchange-like
      // QPQ
      for(auto a=0; a!=nvirt; ++a)
        for(auto j=0; j!=nocc; ++j)
          for(auto b=0; b!=nvirt; ++b)
            Fsscf_e(bf1, bf2) += Vmajb[bf1*najb+a*nocc*nvirt+j*nvirt+b] *
              Vmajb[bf2*najb+b*nocc*nvirt+j*nvirt+a];
      // PQP
      for(auto i=0; i!=nocc; ++i)
        for(auto j=0; j!=nocc; ++j)
          for(auto b=0; b!=nvirt; ++b)
            Fsscf_e(bf1, bf2) -= Vinjb[bf1*nijb+i*nocc*nvirt+j*nvirt+b] *
              Vinjb[bf2*nijb+j*nocc*nvirt+i*nvirt+b];
      Fsscf_e(bf2, bf1) = Fsscf_e(bf1, bf2);
    }
#ifdef TIMING
  tend = std::chrono::high_resolution_clock::now();
  time_elapsed = tend - tstart;
  std::cout << "time-forming Fsscf: " << time_elapsed.count() << std::endl;
#endif

  return 2.*Fsscf_c - Fsscf_e;
}

// A O(N^5) time, O(N^3) memory implementation of sigma-SCF Fock builder
// The storage of intermediate tensors Vmajb and Vmiaj are avoided here and
// the O(N^3) storage comes from intermediate tensor Vmnjb w/ integral screening
Matrix compute_2body_fock_sscf_n3mem(const BasisSet& obs, const Matrix& C_occ,
                                     const Matrix& C_virt)
{
  const auto n = obs.nbf();
  const auto nocc = C_occ.cols();
  const auto nvirt = C_virt.cols();

#ifdef TIMING
  auto tstart = std::chrono::high_resolution_clock::now();
#endif
  // Vmnjb scales N^2*No*Nv ~ N^4 at this point because shell pair-based
  // integral screening is not implemented
  // Once the screening is done, Vmnjb scales as Nsp*No*Nv ~ N^3 for large N.
  std::vector<double> Vmnjb(n*n*nocc*nvirt, 0.);
  compute_Vmnjb_prmt(obs, C_occ, C_virt, Vmnjb);
#ifdef TIMING
  auto tend = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_elapsed = tend - tstart;
  std::cout << "time-forming Vmnjb: " << time_elapsed.count() << std::endl;
#endif

#ifdef TIMING
  tstart = std::chrono::high_resolution_clock::now();
#endif
  Matrix Fsscf = Matrix::Zero(n,n);
  // building the QPQ term
  for(auto j=0; j!=nocc; ++j)
    for(auto a=0; a!=nvirt; ++a) {

      Matrix X(n,n);  // X(mu,nu) = (mu,nu|j,a)
      for(auto bf1=0; bf1!=n; ++bf1)
        for(auto bf2=0; bf2<=bf1; ++bf2)
          X(bf1,bf2) = X(bf2,bf1) =
            Vmnjb[bf1*n*nocc*nvirt+bf2*nocc*nvirt+j*nvirt+a];

      // two-fold symmetry is used here
      for(auto b=0; b<=a; ++b) {

        Matrix x = X * C_virt.col(b);  // x(mu) = (mu,b|j,a)

        if (a == b) {
          for(auto bf1=0; bf1!=n; ++bf1)
            for(auto bf2=0; bf2<=bf1; ++bf2)
              Fsscf(bf1,bf2) += x(bf1,0) * x(bf2,0);
        } else {
          Matrix Y(n,n);  // Y(mu,nu) = (mu,nu|j,b)
          for(auto bf1=0; bf1!=n; ++bf1)
            for(auto bf2=0; bf2<=bf1; ++bf2)
              Y(bf1,bf2) = Y(bf2,bf1) =
                Vmnjb[bf1*n*nocc*nvirt+bf2*nocc*nvirt+j*nvirt+b];

          Matrix y = Y * C_virt.col(a);  // y(mu) = (nu,a|j,b)

          for(auto bf1=0; bf1!=n; ++bf1)
            for(auto bf2=0; bf2<=bf1; ++bf2)
              Fsscf(bf1,bf2) += 2. * (x(bf1,0)*x(bf2,0) + y(bf1,0)*y(bf2,0)) -
                (x(bf1,0)*y(bf2,0) + y(bf1,0)*x(bf2,0));
        }
      }
    }
#ifdef TIMING
  tend = std::chrono::high_resolution_clock::now();
  time_elapsed = tend - tstart;
  std::cout << "time-forming QPQ : " << time_elapsed.count() << std::endl;
#endif

#ifdef TIMING
  tstart = std::chrono::high_resolution_clock::now();
#endif
  // building the PQP term
  for(auto a=0; a!=nvirt; ++a)
    for(auto i=0; i!=nocc; ++i) {

      Matrix X(n,n);  // (mu,nu|i,a)
      for(auto bf1=0; bf1!=n; ++bf1)
        for(auto bf2=0; bf2<=bf1; ++bf2)
          X(bf1,bf2) = X(bf2,bf1) =
            Vmnjb[bf1*n*nocc*nvirt+bf2*nocc*nvirt+i*nvirt+a];

      // two-fold symmetry is used here
      for(auto j=0; j<=i; ++j) {

        Matrix x = X * C_occ.col(j);  // (mu,j|i,a)

        if (i == j) {
          for(auto bf1=0; bf1!=n; ++bf1)
            for(auto bf2=0; bf2<=bf1; ++bf2)
              Fsscf(bf1,bf2) -= x(bf1,0) * x(bf2,0);
        } else {
          Matrix Y(n,n);  // (mu,nu|j,a)
          for(auto bf1=0; bf1!=n; ++bf1)
            for(auto bf2=0; bf2<=bf1; ++bf2)
              Y(bf1,bf2) = Y(bf2,bf1) =
                Vmnjb[bf1*n*nocc*nvirt+bf2*nocc*nvirt+j*nvirt+a];

          Matrix y = Y * C_occ.col(i);  // (mu,i|j,a)

          for(auto bf1=0; bf1!=n; ++bf1)
            for(auto bf2=0; bf2<=bf1; ++bf2)
              Fsscf(bf1,bf2) += -2. * (x(bf1,0)*x(bf2,0) + y(bf1,0)*y(bf2,0)) +
                x(bf1,0)*y(bf2,0) + y(bf1,0)*x(bf2,0);
        }
      }
    }
#ifdef TIMING
  tend = std::chrono::high_resolution_clock::now();
  time_elapsed = tend - tstart;
  std::cout << "time-forming PQP : " << time_elapsed.count() << std::endl;
#endif

  // scattering lower triangular to upper triangular
  for(auto bf1=0; bf1!=n; ++bf1)
    for(auto bf2=0; bf2<bf1; ++bf2)
      Fsscf(bf2,bf1) = Fsscf(bf1,bf2);

  return Fsscf;
}

// A O(N^5) time, O(N^2) memory implementation of sigma-SCF Fock builder
// The storage of intermediate tensors Vmnjb are avoided here
// by moving the loop over some index to the outmost
// This reduces the memory requirement to at most Nsp*Nv ~ N2
// But the down side is that integrals need to be computed multiple times
// which slows them down.
// There are several ways to improve this.
// 1. split the outmost loop to m small loops, which increases the memory
//    requirement to m*Nsp*Nv, but also reduces the time to compute integrals
//    to 1/m.
// 2. density fitting, which could also achieve the O(N^2) storage.
Matrix compute_2body_fock_sscf_n2mem(const BasisSet& obs, const Matrix& C_occ,
                                     const Matrix& C_virt)
{
  using libint2::Shell;
  using libint2::Engine;
  using libint2::Operator;

  const auto n = obs.nbf();
  const auto nocc = C_occ.cols();
  const auto nvirt = C_virt.cols();
  const auto nshells = obs.size();

  // construct the electron repulsion integrals engine
  Engine engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);

  auto shell2bf = obs.shell2bf();

  // buf[0] points to the target shell set after every call  to engine.compute()
  const auto& buf = engine.results();

  // initialize intermediate tensor. note that j is fixed so it scales
  // O(N^2*Nvirt) w/o integral-screening and O(Nsp*Nvirt) after screening.
  std::vector<double> Vmnjb(n*n*nvirt, 0.);

#ifdef TIMING
  auto tstart = std::chrono::high_resolution_clock::now();
  auto tend = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_elapsed_int = tend - tstart;
  std::chrono::duration<double> time_elapsed_mm = tend - tstart;
  std::chrono::duration<double> time_elapsed_xf = tend - tstart;
#endif

  Matrix Fsscf_c = Matrix::Zero(n,n);
  Matrix Fsscf_e = Matrix::Zero(n,n);
  for(auto i=0; i!=nocc; ++i) {

    // s[i] indices the i-th shell;
    // bf[i] points to the first basis func in that shell
    // four-fold symmetry is used
    for(auto s1=0; s1!=nshells; ++s1) {

      auto bf1_first = shell2bf[s1];
      auto n1 = obs[s1].size();

      for(auto s2=0; s2<=s1; ++s2) {

        auto bf2_first = shell2bf[s2];
        auto n2 = obs[s2].size();

        // could reduce this storage to 1/2 but ignored here
        std::vector<double> tmp4(n1*n2*n*n, 0.);

#ifdef TIMING
        tstart = std::chrono::high_resolution_clock::now();
#endif
        for(auto s3=0; s3!=nshells; ++s3) {

          auto bf3_first = shell2bf[s3];
          auto n3 = obs[s3].size();

          for(auto s4=0; s4<=s3; ++s4) {

            auto bf4_first = shell2bf[s4];
            auto n4 = obs[s4].size();

            engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
            const auto* buf_1234 = buf[0];
            if (buf_1234 == nullptr)
              continue; // if all integrals screened out, skip to next quartet

            // brute-force assign quartet-ERI;
            // could be improved by some tensor library say Tiled Array
            for(auto f1=0, f1234=0; f1!=n1; ++f1) {
              for(auto f2=0; f2!=n2; ++f2) {
                for(auto f3=0; f3!=n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for(auto f4=0; f4!=n4; ++f4, ++f1234) {
                    const auto bf4 = f4 + bf4_first;
                    tmp4[f1*n2*n*n+f2*n*n+bf3*n+bf4] =
                      tmp4[f1*n2*n*n+f2*n*n+bf4*n+bf3] = buf_1234[f1234];
                  } // f4
                } // f3
              } // f2
            } // f1

          } // s4
        } // s3
#ifdef TIMING
        tend = std::chrono::high_resolution_clock::now();
        time_elapsed_int += std::chrono::duration_cast<
          std::chrono::duration<double>>(tend - tstart);

        tstart = std::chrono::high_resolution_clock::now();
#endif
        Matrix tmp2(n, n);
        for(auto f1=0; f1!=n1; ++f1) {
          const auto bf1 = f1 + bf1_first;
          for(auto f2=0; f2!=n2; ++f2) {
            const auto bf2 = f2 + bf2_first;
            for(auto bf3=0; bf3!=n; ++bf3)
              for(auto bf4=0; bf4<=bf3; ++bf4)
                tmp2(bf3,bf4) = tmp2(bf4,bf3) =
                  tmp4[f1*n2*n*n+f2*n*n+bf3*n+bf4];

            Matrix tmp1 = C_occ.col(i).transpose() * tmp2 * C_virt;

            for(auto b=0; b!=nvirt; ++b)
              Vmnjb[bf1*n*nvirt+bf2*nvirt+b] =
                Vmnjb[bf2*n*nvirt+bf1*nvirt+b] = tmp1(0,b);
          } // f2
        } // f1
#ifdef TIMING
        tend = std::chrono::high_resolution_clock::now();
        time_elapsed_mm += std::chrono::duration_cast<
          std::chrono::duration<double>>(tend - tstart);
#endif
      } // s2
    } // s1

#ifdef TIMING
    tstart = std::chrono::high_resolution_clock::now();
#endif
    for(auto a=0; a!=nvirt; ++a) {

      Matrix tmp21(n,n);  // tmp21(mu,nu) = (mu,nu|j,a)
      for(auto bf1=0; bf1!=n; ++bf1)
        for(auto bf2=0; bf2<=bf1; ++bf2)
          tmp21(bf1,bf2) = tmp21(bf2,bf1) = Vmnjb[bf1*n*nvirt+bf2*nvirt+a];

      for(auto b=0; b!=nvirt; ++b) {

        Matrix tmp22(n,n);  // tmp22(mu,nu) = (mu,nu|j,b)
        for(auto bf1=0; bf1!=n; ++bf1)
          for(auto bf2=0; bf2<=bf1; ++bf2)
            tmp22(bf1,bf2) = tmp22(bf2,bf1) = Vmnjb[bf1*n*nvirt+bf2*nvirt+b];

        Matrix tmp11 = tmp21*C_virt.col(b);  // (mu,b|j,a)
        Matrix tmp12 = tmp22*C_virt.col(a);  // (mu,a|j,b)
        // std::cout << "norm(tmp11 - tmp12) = " << (tmp11-tmp12).norm() << std::endl;

        for(auto bf1=0; bf1!=n; ++bf1)
          for(auto bf2=0; bf2<=bf1; ++bf2) {
            Fsscf_c(bf1,bf2) += tmp11(bf1,0)*tmp11(bf2,0);
            Fsscf_e(bf1,bf2) += tmp11(bf1,0)*tmp12(bf2,0);
          } // bf2

      } // b
    } // a
#ifdef TIMING
    tend = std::chrono::high_resolution_clock::now();
    time_elapsed_xf += std::chrono::duration_cast<
      std::chrono::duration<double>>(tend - tstart);
#endif
  } // i

  for(auto bf1=0; bf1!=n; ++bf1)
    for(auto bf2=0; bf2<bf1; ++bf2) {
      Fsscf_c(bf2,bf1) = Fsscf_c(bf1,bf2);
      Fsscf_e(bf2,bf1) = Fsscf_e(bf1,bf2);
    }

#ifdef TIMING
  printf("  time-int: %10.5f\n", time_elapsed_int.count());
  printf("  time-mm : %10.5f\n", time_elapsed_mm.count());
  printf("  time-xf : %10.5f\n", time_elapsed_xf.count());
#endif
  // dumpMat(Fsscf_c, "% .5f", "Fsscf_c");
  // dumpMat(Fsscf_e, "% .5f", "Fsscf_e");

  return 2.*Fsscf_c - Fsscf_e;
}
