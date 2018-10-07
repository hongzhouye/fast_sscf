EIGEN_ROOT = /Users/hzye/local/opt/eigen-3.2.8/include/eigen3
LIBINT_ROOT = /Users/hzye/local/opt/libint-2.4.2

CXX = g++-8
CXXFLAGS = -O2 -std=c++11 -w $(extra)
CXXINCS = -I$(LIBINT_ROOT)/include -I$(LIBINT_ROOT)/libint2 -Ieri -Ihartree-fock -DPREP_LIBINT2_SKIP_BOOST -DSRCDATADIR=\"$(LIBINT_ROOT)/lib/basis\" -I$(EIGEN_ROOT)
CXXLIBS = -L$(LIBINT_ROOT)/lib -lint2


sigma-scf.out: main.o sscf_utils.o
	$(CXX) $(CXXFLAGS) $(CXXINCS) $(CXXLIBS) \
		main.o sscf_utils.o -o sigma-scf.out

main.o: main.cc sscf_utils.h
	$(CXX) $(CXXFLAGS) $(CXXINCS) $(CXXLIBS) main.cc -c

sscf_utils.o: sscf_utils.cc sscf_utils.h
	$(CXX) $(CXXFLAGS) $(CXXINCS) $(CXXLIBS) sscf_utils.cc -c

clean:
	rm -f *.o *.out
