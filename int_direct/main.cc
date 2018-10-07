/*
 *  A comparison of different implementation of integral-direct sigma-SCF Fock
 *  matrix builder:
 *    1. AO-basis O(K^5) algorithm, brute-force
 *    2. AO-basis O(K^5) algorithm, exploit permutational symmetry
 *    3. MO-basis O(K^4) algorithm
 *    4. MO-basis O(K^4) algorithm with Schwarz screening
 *    5. MO-basis O(K^4) algorithm with Schwarz screening and OMP parallel
 */


// helper funcitions
#include "sscf_utils.h"


int main(int argc, char* argv[])
{
    using std::cout;
    using std::cerr;
    using std::endl;

    try {
      /*** =========================== ***/
      /*** initialize molecule         ***/
      /*** =========================== ***/

      // read geometry from a file
      if (argc < 3) throw "Usage: xyz basis";
      const auto filename = argv[1];
      const auto basisname = argv[2];
      bool do_density_fitting = false;
      std::vector<Atom> atoms = read_geometry(filename);

      cout << "Atomic Cartesian coordinates (a.u.):" << endl;
      for (const auto& a : atoms)
        std::cout << a.atomic_number << " " << a.x << " " << a.y << " " << a.z
                  << std::endl;

      // count the number of electrons
      const auto nelectron = count_nelectron(atoms);
      const auto ndocc = nelectron / 2;
      cout << "# of electrons = " << nelectron << endl;

      // compute the nuclear repulsion energy
      const auto enuc = compute_enuc(atoms);
      cout << "Nuclear repulsion energy = " << std::setprecision(15) << enuc
           << endl;

      BasisSet obs(basisname, atoms);
      auto nao = obs.nbf();
      auto nvirt = nao - ndocc;
      cout << "orbital basis set rank = " << nao << endl;

      /*** =========================== ***/
      /*** compute 1-e integrals       ***/
      /*** =========================== ***/

      // initializes the Libint integrals library ... now ready to compute
      libint2::initialize();

      // compute overlap integrals
      auto S = compute_1body_ints(obs, Operator::overlap);
      // dumpMat(S, "% .5f", "\n\tOverlap Integrals:");

      // compute kinetic-energy integrals
      auto T = compute_1body_ints(obs, Operator::kinetic);
      // dumpMat(T, "% .5f", "\n\tKinetic-Energy Integrals:");

      // compute nuclear-attraction integrals
      Matrix V = compute_1body_ints(obs, Operator::nuclear, atoms);
      // dumpMat(V, "% .5f", "\n\tNuclear Attraction Integrals:");

      // Core Hamiltonian = T + V
      Matrix H = T + V;
      // dumpMat(H, "% .5f", "\n\tCore Hamiltonian:");

      // T and V no longer needed, free up the memory
      T.resize(0,0);
      V.resize(0,0);

      // compute orthogonalizer X such that X.transpose() . S . X = I
      Matrix X, Xinv;
      double XtX_condition_number;  // condition number of "re-conditioned"
                                    // overlap obtained as Xinv.transpose() . Xinv
      // one should think of columns of Xinv as the conditioned basis
      // Re: name ... cond # (Xinv.transpose() . Xinv) = cond # (X.transpose() .
      // X)
      // by default assume can manage to compute with condition number of S <=
      // 1/eps
      // this is probably too optimistic, but in well-behaved cases even 10^11 is
      // OK
      double S_condition_number_threshold =
          1.0 / std::numeric_limits<double>::epsilon();
      std::tie(X, Xinv, XtX_condition_number) =
          conditioning_orthogonalizer(S, S_condition_number_threshold);

      /*** =========================== ***/
      /*** build initial-guess density ***/
      /*** =========================== ***/

      Matrix C = compute_soad_from_minbs(atoms, obs, H, X, ndocc);
      Matrix C_occ = C.leftCols(ndocc);
      Matrix C_virt = C.rightCols(nvirt);
      Matrix D = C_occ * C_occ.transpose();
      Matrix Q = C_virt * C_virt.transpose();
      // dumpMat(D, "% .5f", "\n\tInitial Density Matrix:");
      cout << "\ntr(P*S) = " << (D*S).trace() << endl;

      Matrix evals;

      /*** =========================== ***/
      /***          SCF loop           ***/
      /*** =========================== ***/

      const auto maxiter = 100;
      const auto conv = 1e-8;
      auto iter = 0;
      auto ehf = 0.0;
      auto ediff_rel = 0.0;
      auto rms_error = 0.0;
      auto n2 = D.cols() * D.rows();
      libint2::DIIS<Matrix> diis(2);  // start DIIS on second iteration

      std::chrono::duration<double> time_fock_tot;

      auto var = 0.0;
      auto vardiff_rel = 0.0;

      do {
        ++iter;

        // Save a copy of the energy and the density
        auto ehf_last = ehf;
        auto var_last = var;

        // build a new Fock matrix
        auto F = H;
        F += compute_2body_fock(obs, D);

        const auto tstart = std::chrono::high_resolution_clock::now();
        // build a sigma-SCF Fock matrix
        Matrix Fsscf = F * (Q-D) * F;
        const Matrix Dt = D * F * Q + Q * F * D;
        Fsscf += compute_2body_fock(obs, Dt);
        // Fsscf += compute_2body_fock_sscf_n4mem(obs, C_occ, C_virt);
        Fsscf += compute_2body_fock_sscf_n3mem(obs, C_occ, C_virt);
        // Fsscf += compute_2body_fock_sscf_n2mem(obs, C_occ, C_virt);
        // Fsscf -= compute_2body_fock_sscf_n2mem(obs, C_virt, C_occ);
        const auto tstop = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> time_elapsed = tstop - tstart;
        if (iter == 1)
        {
          time_fock_tot = std::chrono::duration_cast<
            std::chrono::duration<double>>(time_elapsed);
          printf("Count = %10.5lf\n", time_fock_tot.count());
        }
        else
          time_fock_tot += std::chrono::duration_cast<
            std::chrono::duration<double>>(time_elapsed);

        // compute HF energy with the non-extrapolated Fock matrix
        ehf = D.cwiseProduct(H + F).sum();
        ediff_rel = std::abs((ehf - ehf_last) / ehf);

        var = (D * F * Q * F).trace() * 2.;
        var += compute_sscf_variance_mo(obs, C_occ, C_virt);
        vardiff_rel = std::abs((var - var_last) / var);

        // compute SCF error
        // Matrix FD_comm = F * D * S - S * D * F;
        Matrix FD_comm = Fsscf * D * S - S * D * Fsscf;
        rms_error = FD_comm.norm() / n2;

        // DIIS extrapolate F
        // Matrix F_diis = F;  // extrapolated F cannot be used in incremental Fock
        Matrix F_diis = Fsscf;  // extrapolated F cannot be used in incremental Fock
                            // build; only used to produce the density
                            // make a copy of the unextrapolated matrix
        diis.extrapolate(F_diis, FD_comm);

        // solve F C = e S C by (conditioned) transformation to F' C' = e C',
        // where
        // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
        Eigen::SelfAdjointEigenSolver<Matrix> eig_solver(X.transpose() *
          F_diis * X);
        evals = eig_solver.eigenvalues();
        C = X * eig_solver.eigenvectors();

        // compute density, D = C(occ) . C(occ)T
        C_occ = C.leftCols(ndocc);
        C_virt = C.rightCols(nvirt);
        D = C_occ * C_occ.transpose();
        Q = C_virt * C_virt.transpose();

        if (iter == 1)
          std::cout << "\n\nIter         E(HF)                 D(E)/E         "
                       "RMS([F,D])/nn       Time(s)\n";
        printf(" %02d %17.9f %15.7f %13.5e %13.5e %10.5lf\n", iter, ehf + enuc,
               var, vardiff_rel, rms_error, time_elapsed.count());

      } while (((vardiff_rel > conv) || (rms_error > conv)) && (iter < maxiter));

      double tfock = time_fock_tot.count() / float(iter);
      printf("** t(Fock build)/cycle = %20.12f\n", tfock);
      printf("** Hartree-Fock energy = %20.12f\n", ehf + enuc);
    }

    catch (const char* ex) {
      cerr << "caught exception: " << ex << endl;
      return 1;
    } catch (std::string& ex) {
      cerr << "caught exception: " << ex << endl;
      return 1;
    } catch (std::exception& ex) {
      cerr << ex.what() << endl;
      return 1;
    } catch (...) {
      cerr << "caught unknown exception\n";
      return 1;
    }
}
