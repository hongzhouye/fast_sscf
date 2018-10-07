EIGEN_ROOT = /Users/hzye/local/opt/eigen-3.2.8/include/eigen3
LIBINT_ROOT = /Users/hzye/local/opt/libint-2.4.2

CXX = g++-8
CXXFLAGS = -O2 -std=c++11 -w
CXXINCS = -I$(LIBINT_ROOT)/include -I$(LIBINT_ROOT)/libint2 -Ieri -Ihartree-fock -DPREP_LIBINT2_SKIP_BOOST -DSRCDATADIR=\"$(LIBINT_ROOT)/lib/basis\" -I$(EIGEN_ROOT)
CXXLIBS = -L$(LIBINT_ROOT)/lib -lint2

hartree-fock.out: hartree-fock.cc
	$(CXX) $(CXXFLAGS) $(CXXINCS) $(CXXLIBS) hartree-fock.cc -o hartree-fock.out
