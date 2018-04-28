%module vstRender
%{
#define SWIG_FILE_WITH_INIT
#include "vstRender.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (float* INPLACE_ARRAY1, int DIM1) {(float* buffer, int sampleLength)};
%include "vstRender.h"
