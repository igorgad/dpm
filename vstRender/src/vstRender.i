%module vstRender

%include <std_pair.i>
%include <std_vector.i>
%include <std_string.i>
%template() std::pair<int,float>;
%template(PluginParams) std::vector<std::pair<int,float> >;

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
