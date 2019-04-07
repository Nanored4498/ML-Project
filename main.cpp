#include <iostream>
#include <fstream>
#include "stb_image_write.h"
#include "filter.hpp"

using namespace std;

typedef unsigned char uchar;


int read_int(ifstream &file) {
	char buf[4];
	file.read(buf, 4);
	int res = 0;
	for(int i = 0; i < 4; i++) {
		res <<= 8;
		res += (uchar) buf[i];
	}
	return res;
}

// Normalize an image
double normalize(uchar *im, double *mu, int size) {
	double sum = 0;
	for(int i = 0; i < size; i++)
		mu[i] = min(255, im[i] + 1), sum += mu[i];
	double mul = size / sum;
	for(int i = 0; i < size; i++)
		mu[i] *= mul;
	return mul;
}

void normalize_array(double *in, double *out, int size) {
	double sum_in = 0, sum_out = 0;
	#pragma omp parallel for
	for(int i = 0; i < size; i++)
		sum_in += in[i], sum_out += out[i];
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
	// double scale0 = normalize(im0, mu0, size);
	// double scale1 = normalize(im1, mu1, size);
	if(!av) delete[] av;
	if(!aw) delete[] aw;
	av = new double[size];
	aw = new double[size];
	for(int i = 0; i < size; i++)
		aw[i] = a;

	double gamma = sigma * sigma;
	vector<complex<double>> filter_col, filter_row;
	gauss_filter(H, filter_col, gamma);
	gauss_filter(W, filter_row, gamma);
	double *conv = new double[size];

	for(int i = 0; i < num_it; i++) {
		double_filter_1D(aw, conv, W, H, filter_col, filter_row);
		normalize_array(aw, conv, size);
		double sum = 0, sum2 = 0;
		for(int j = 0; j < size; j++)
			sum += aw[j], sum2 += conv[j];
		cout << "A " << aw[0] << " " << sum << " " << sum2 << endl;

		sum = 0, sum2 = 0;
		for(int j = 0; j < size; j++)
			sum += av[j];
		#pragma omp parallel for
		for(int j = 0; j < size; j++)
			av[j] = a * mu0[j] / conv[j];
		for(int j = 0; j < size; j++)
			sum2 += av[j];
		cout << "B " << av[0] << " " << sum << " " << sum2 << endl;


		double_filter_1D(av, conv, W, H, filter_col, filter_row);
		normalize_array(av, conv, size);
		// sum = 0, sum2 = 0;
		// for(int j = 0; j < size; j++)
		// 	sum += v[j], sum2 += conv[j];
		// cout << "C " << sum << " " << sum2 << endl;

		// sum = 0, sum2 = 0;
		// for(int j = 0; j < size; j++)
		// 	sum += w[j];
		#pragma omp parallel for
		for(int j = 0; j < size; j++)
			aw[j] = a * mu1[j] / conv[j];
		// for(int j = 0; j < size; j++)
		// 	sum2 += w[j];
		// cout << "D " << sum << " " << sum2 << endl;
	}

	// uchar* im = new uchar[size];
	// double_filter_1D(w, conv, W, H, filter_col, filter_row);
	// normalize_array(w, conv, size);
	// for(int i = 0; i < size; i++)
	// 	im[i] = min(255.0, conv[i]*v[i]*size/scale0);
	// stbi_write_png("mu0b.png", W, H, 1, im, 0);
	// double_filter_1D(v, conv, W, H, filter_col, filter_row);
	// normalize_array(v, conv, size);
	// for(int i = 0; i < size; i++)
	// 	im[i] = min(255.0, conv[i]*w[i]*size/scale1);
	// stbi_write_png("mu1.png", W, H, 1, im, 0);
	// delete[] im;

	delete[] mu0;
	delete[] mu1;
	delete[] conv;

	double res = 0;
	for(int i = 0; i < size; i++)
		res += mu0[i] * log(av[i] / a) + mu1[i] * log(aw[i] / a);
	return gamma * a * res;
}

uchar* barycenter(vector<uchar*> ims, vector<double> alpha, int W, int H, double sigma, int num_it) {
	int k = ims.size();
	if((int) alpha.size() < k)
		return nullptr;
	int size = W*H;
	double a = 1.0 / size;
	vector<double*> mus;
	vector<double> scales;
	vector<double*> avs, aws, ds;
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
	vector<complex<double>> filter_col, filter_row;
	gauss_filter(H, filter_col, gamma);
	gauss_filter(W, filter_row, gamma);
	double *conv = new double[size];
	double *mu = new double[size];

	for(int i = 0; i < num_it; i++) {
		for(int j = 0; j < size; j++)
			mu[j] = 1;

		// Project onto C1
		for(int j = 0; j < k; j++) {
			double_filter_1D(avs[j], conv, W, H, filter_col, filter_row);
			normalize_array(avs[j], conv, size);
			#pragma omp parallel for
			for(int l = 0; l < size; l++)
				aws[j][l] = a * mus[j][l] / conv[l];
			
			double_filter_1D(aws[j], conv, W, H, filter_col, filter_row);
			normalize_array(aws[j], conv, size);
			#pragma omp parallel for
			for(int l = 0; l < size; l++) {
				ds[j][l] = avs[j][l] / a * conv[l];
				mu[l] *= pow(ds[j][l], alpha[j]);
			}
		}

		// Sharpening
		// Later

		// Project onto C2
		for(int j = 0; j < k; j++)
			#pragma omp parallel for
			for(int l = 0; l < size; l++)
				avs[j][l] *= mu[l] / ds[j][l];
	}

	uchar* res = new uchar[size];
	double scale = 0;
	for(int i = 0; i < k; i++)
		scale += alpha[i] * scales[i];
	for(int i = 0; i < size; i++)
		res[i] = min(255.0, mu[i] / scale);
	return res;
}

int main() {
	ifstream file_im("train_images");
	ifstream file_lab("train_labels");

	read_int(file_im);
	int num_images = read_int(file_im);
	cout << "number of images: " << num_images << endl;
	int W = read_int(file_im);
	int H = read_int(file_im);

	read_int(file_lab);
	read_int(file_lab);
	char lab[1];
	file_lab.read(lab, 1);
	cout << (int) *lab << endl;

	uchar *im0 = new uchar[W*H];
	file_im.read((char*) im0, W*H);
	stbi_write_png("out.png", W, H, 1, im0, 0);

	double *v = 0, *w = 0;
	uchar *im1 = new uchar[W*H];
	for(int i = 0; i < 14; i++) {
		file_im.read((char*) im1, W*H);
		file_lab.read(lab, 1);
		cout << (int) *lab << " " << wasserstein(im0, im1, W, H, 1.5, 10, v, w) << endl;
	}
	uchar* bar = barycenter({im0, im1}, {0.5, 0.5}, W, H, 1.5, 8);
	stbi_write_png("bar.png", W, H, 1, bar, 0);

	free(im0);
	free(im1);
	delete[] v;
	delete[] w;
	delete[] bar;
	return 0;
}