#include "wasserstein.hpp"
#include "filter.hpp"
#include <iostream>
#include "stb_image_write.h"

typedef unsigned char uchar;

// Normalize an image
double normalize(uchar *im, double *mu, int size) {
	double sum = 0;
	for(int i = 0; i < size; i++) {
		mu[i] = std::min(255, im[i] + 1);
		sum += mu[i];
	}
	double mul = size / sum;
	for(int i = 0; i < size; i++)
		mu[i] *= mul;
	return mul;
}

void normalize_conv(double *in, double *out, int size) {
	double sum_in = 0, sum_out = 0;
	#pragma omp parallel for reduction(+: sum_in, sum_out)
	for(int i = 0; i < size; i++) {
		out[i] = std::max(1e-10, out[i]);
		sum_in += in[i];
		sum_out += out[i];
	}
	#define ALPHA 0.5
	double scale = ALPHA + (1-ALPHA) * sum_in / sum_out;
	#pragma omp parallel for
	for(int i = 0; i < size; i++)
		out[i] *= scale;
}

double wasserstein(uchar *im0, uchar *im1, int W, int H, double sigma, int num_it,
				double *av, double *aw) {
	int size = W*H;
	double a = 1.0 / size;
	double *mu0 = new double[size];
	double *mu1 = new double[size];
	normalize(im0, mu0, size);
	normalize(im1, mu1, size);
	if(!av) delete[] av;
	if(!aw) delete[] aw;
	av = new double[size];
	aw = new double[size];
	for(int i = 0; i < size; i++)
		aw[i] = a;

	double gamma = sigma * sigma;
	std::vector<std::complex<double>> filter_col, filter_row;
	gauss_filter(H, filter_col, gamma);
	gauss_filter(W, filter_row, gamma);
	double *conv = new double[size];

	double res = 0;

	for(int i = 0; i < num_it; i++) {
		double_filter_1D(aw, conv, W, H, filter_col, filter_row);
		normalize_conv(aw, conv, size);
		#pragma omp parallel for
		for(int j = 0; j < size; j++)
			av[j] = a * mu0[j] / conv[j];


		double_filter_1D(av, conv, W, H, filter_col, filter_row);
		normalize_conv(av, conv, size);
		#pragma omp parallel for
		for(int j = 0; j < size; j++)
			aw[j] = a * mu1[j] / conv[j];

		if(i == num_it-2) {
			res = 0;
			#pragma omp parallel for reduction(+:res)
			for(int j = 0; j < size; j++)
				res += mu0[j] * std::log(av[j] / a) + mu1[j] * std::log(aw[j] / a);
			res *= gamma * a;
		} else if(i == num_it-1) {
			double new_res = 0;
			#pragma omp parallel for reduction(+:new_res)
			for(int j = 0; j < size; j++)
				new_res += mu0[j] * std::log(av[j] / a) + mu1[j]* std::log(aw[j] / a);
			new_res *= gamma * a;
			bool redo = std::abs(new_res - res) > 1e-2;
			res = new_res;
			if(redo) num_it ++;
		}
	}

	// uchar* im = new uchar[size];
	// double_filter_1D(aw, conv, W, H, filter_col, filter_row);
	// normalize_conv(aw, conv, size);
	// for(int i = 0; i < size; i++)
	// 	im[i] = std::min(255.0, conv[i]*av[i]/a/scale0);
	// stbi_write_png("mu0b.png", W, H, 1, im, 0);
	// double_filter_1D(av, conv, W, H, filter_col, filter_row);
	// normalize_conv(av, conv, size);
	// for(int i = 0; i < size; i++)
	// 	im[i] = std::min(255.0, conv[i]*aw[i]/a/scale1);
	// stbi_write_png("mu1b.png", W, H, 1, im, 0);
	// delete[] im;

	delete[] mu0;
	delete[] mu1;
	delete[] conv;

	return res;
}

double entropy(double *mu, double a, int size) {
	double res = 0;
	#pragma omp parallel for reduction(+:res)
	for(int i = 0; i < size; i++)
		res += mu[i] * (std::log(mu[i]) - 1);
	return - res * a;
}

double entropy_prime(double *mu, double a, int size, double beta) {
	double res = 0;
	#pragma omp parallel for reduction(+:res)
	for(int i = 0; i < size; i++)
		res += mu[i] * std::pow(std::log(mu[i]), 2.0);
	return - res * a / beta;
}

void entropic_sharpening(double *mu, double H0, double a, int size) {
	double H = entropy(mu, a, size);
	if(H > H0) {
		double beta = 1.0;
		while(std::abs(H-H0) > 2e-3) {
			double prime = entropy_prime(mu, a, size, beta);
			double beta2 = beta + (H0 - H) / prime;
			double power = beta2 / beta;
			beta = beta2;
			#pragma omp parallel for
			for(int i = 0; i < size; i++)
				mu[i] = std::pow(mu[i], power);
			H = entropy(mu, a, size);
		}
	}
}

uchar* barycenter(std::vector<uchar*> ims, std::vector<double> alpha, int W, int H, double sigma, int num_it) {
	int k = ims.size();
	if((int) alpha.size() < k)
		return nullptr;
	int size = W*H;
	double a = 1.0 / size;
	std::vector<double*> mus;
	std::vector<double> scales;
	std::vector<double*> avs, aws, ds;
	for(int i = 0; i < k; i++) {
		double *mu = new double[size];
		double scale = normalize(ims[i], mu, size);
		mus.push_back(mu);
		scales.push_back(scale);
		double *av = new double[size];
		double *aw = new double[size];
		double *d = new double[size];
		for(int j = 0; j < size; j++)
			av[j] = a;
		avs.push_back(av);
		aws.push_back(aw);
		ds.push_back(d);
	}

	double gamma = sigma * sigma;
	std::vector<std::complex<double>> filter_col, filter_row;
	gauss_filter(H, filter_col, gamma);
	gauss_filter(W, filter_row, gamma);
	double *conv = new double[size];
	double *mu = new double[size];
	double *mu2 = new double[size];

	double H0 = std::log(a);
	for(int i = 0; i < k; i++)
		H0 = std::max(H0, entropy(mus[i], a, size));
	double delta_conv = double(W * H) / 1000.0;

	for(int i = 0; i < num_it; i++) {
		for(int j = 0; j < size; j++)
			mu[j] = 1;
		// Project onto C1
		for(int j = 0; j < k; j++) {
			double_filter_1D(avs[j], conv, W, H, filter_col, filter_row);
			normalize_conv(avs[j], conv, size);
			#pragma omp parallel for
			for(int l = 0; l < size; l++)
				aws[j][l] = a * mus[j][l] / conv[l];
			
			double_filter_1D(aws[j], conv, W, H, filter_col, filter_row);
			normalize_conv(aws[j], conv, size);
			#pragma omp parallel for
			for(int l = 0; l < size; l++) {
				ds[j][l] = avs[j][l] / a * conv[l];
				mu[l] *= pow(ds[j][l], alpha[j]);
			}
		}
		// Sharpening
		entropic_sharpening(mu, H0, a, size);
		// Project onto C2
		for(int j = 0; j < k; j++)
			#pragma omp parallel for
			for(int l = 0; l < size; l++)
				avs[j][l] *= mu[l] / ds[j][l];
		if(i == num_it - 2)
			#pragma omp parallel for
			for(int j = 0; j < size; j++)
				mu2[j] = mu[j];
		if(i == num_it- 1) {
			double sum = 0;
			#pragma omp parallel for reduction(+:sum)
			for(int i = 0; i < size; i++)
				sum += std::abs(mu[i] - mu2[i]), mu2[i] = mu[i];
			if(sum > delta_conv)
				num_it ++, delta_conv += 0.42 + 0.02*(sum - delta_conv);
			std::cout << sum << std::endl;
		}
	}
	uchar* res = new uchar[size];
	double scale = 0;
	for(int i = 0; i < k; i++)
		scale += alpha[i] * scales[i];
	for(int i = 0; i < size; i++)
		res[i] = std::min(255.0, mu[i] / scale);

	return res;
}