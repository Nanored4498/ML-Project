#ifndef DEF_FILTER_H
#define DEF_FILTER_H

#include <vector>
#include <complex>

#define Complex std::complex<double>
#define VC std::vector<Complex>
#define VVC std::vector<VC>

void fft(VC& x);

void ifft(VC& x);

void gauss_filter(int N, VC &filter, double gamma);

void double_filter_1D(double* in, double* out, int W, int H, VC &filter_col, VC &filter_row);

#undef Complex
#undef VC
#undef VCC

#endif