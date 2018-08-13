# Find the CBlas and lapack libraries
#
# It will search MKLML, atlas, OpenBlas, reference-cblas in order.
#
# If any cblas implementation found, the following variable will be set.
#    CBLAS_PROVIDER  # one of MKLML, OPENBLAS, REFERENCE
#    CBLAS_INC_DIR   # the include directory for cblas.
#    CBLAS_LIBS      # a list of libraries should be linked by paddle.
#                    # Each library should be full path to object file.


## Then find openblas.
