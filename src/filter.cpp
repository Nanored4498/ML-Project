#include "filter.hpp"
#include <cmath>

#define Complex std::complex<double>
#define VC std::vector<Complex>
#define VVC std::vector<VC>

void fft(VC& x) {
    int N = x.size();
    if(N <= 1) return;
	int N2 = N >> 1;
	VC even(N2), odd(N2);
	for(int i = 0; i < N2; i++) {
		even[i] = x[i<<1];
		odd[i] = x[(i<<1)+1];
	}
	fft(even);
	fft(odd);
	for(int k = 0; k < N2; k++) {
    	Complex t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k+N2] = even[k] - t;
    }
}

void ifft(VC& x) {
	int N = x.size();
	#pragma omp parallel for
	for(int i = 0; i < N; i++) x[i] = std::conj(x[i]);
	fft(x);
	#pragma omp parallel for
	for(int i = 0; i < N; i++) x[i] = std::conj(x[i]) / ((double) N);
}

void pad(VC& x) {
	int p = 1, N = x.size();
	while(p < N) p *= 2;
	while(N++ < p) x.push_back(0); 
}

void apply_filter_1D(VC &data, VC& filter) {
	int N = data.size();
	pad(data);
	int N2 = data.size();
	fft(data);
	for(int i = 0; i < N2; i++)
		data[i] *= filter[i];
	ifft(data);
	data.resize(N);
}

void gauss_ifilter(int N, VC& filter, double gamma) {
	int p = 1;
	while(p < N) p *= 2;
	N = p;
	double s = 0;
	for(int i = 0; i < N; i++) {
		double v = i < N/2 ? i : N - i;
		v = std::exp(-v*v / gamma);
		s += v;
		filter.push_back(v);
	}
	for(int i = 0; i < N; i++) filter[i] /= s;
}

void gauss_filter(int N, VC &filter, double gamma) {
	filter.clear();
	gauss_ifilter(N, filter, gamma);
	fft(filter);
}

void get_cols(double* in, VVC &cols, int W, int H) {
	for(int i = 0; i < W; i++) {
		VC co;
		for(int j = 0; j < H; j++) co.push_back(in[i + j*W]);
		cols.push_back(co);
	}
}

void cols_to_rows(VVC &cols, VVC &rows, int W, int H) {
	rows.clear();
	for(int i = 0; i < H; i++) {
		VC r;
		for(int j = 0; j < W; j++) r.push_back(cols[j][i]);
		rows.push_back(r);
	}
}

void rows_to_uchar(double* out, VVC rows, int W, int H) {
	#pragma omp parallel for
	for(int i = 0; i < W; i ++)
		for(int j = 0; j < H; j++)
			out[i + j*W] = rows[j][i].real();
}

void double_filter_1D(double* in, double* out, int W, int H, VC &filter_col, VC &filter_row) {
	VVC cols, rows;
	get_cols(in, cols, W, H);
	for(int i = 0; i < W; i++) apply_filter_1D(cols[i], filter_col);
	cols_to_rows(cols, rows, W, H);
	for(int i = 0; i < H; i++) apply_filter_1D(rows[i], filter_row);
	rows_to_uchar(out, rows, W, H);
}