#include <iostream>
#include <fstream>
#include "stb_image_write.h"
#include "stb_image.h"
#include "filter.hpp"
#include <cstring>

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
	for(int i = 0; i < size; i++) {
		mu[i] = min(255, im[i] + 1);
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
		out[i] = max(1e-10, out[i]);
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
		normalize_conv(aw, conv, size);
		// double sum = 0, sum2 = 0;
		// for(int j = 0; j < size; j++)
		// 	sum += aw[j], sum2 += conv[j];
		// cout << "A " << aw[0] << " " << sum << " " << sum2 << endl;

		// sum = 0, sum2 = 0;
		// for(int j = 0; j < size; j++)
		// 	sum += av[j];
		#pragma omp parallel for
		for(int j = 0; j < size; j++)
			av[j] = a * mu0[j] / conv[j];
		// for(int j = 0; j < size; j++)
		// 	sum2 += av[j];
		// cout << "B " << av[0] << " " << sum << " " << sum2 << endl;


		double_filter_1D(av, conv, W, H, filter_col, filter_row);
		normalize_conv(av, conv, size);
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
	// normalize_conv(w, conv, size);
	// for(int i = 0; i < size; i++)
	// 	im[i] = min(255.0, conv[i]*v[i]*size/scale0);
	// stbi_write_png("mu0b.png", W, H, 1, im, 0);
	// double_filter_1D(v, conv, W, H, filter_col, filter_row);
	// normalize_conv(v, conv, size);
	// for(int i = 0; i < size; i++)
	// 	im[i] = min(255.0, conv[i]*w[i]*size/scale1);
	// stbi_write_png("mu1.png", W, H, 1, im, 0);
	// delete[] im;

	double res = 0;
	for(int i = 0; i < size; i++)
		res += mu0[i] * log(av[i] / a) + mu1[i] * log(aw[i] / a);

	delete[] mu0;
	delete[] mu1;
	delete[] conv;

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
	for(int i = 0; i < size; i++) {
		res[i] = min(255.0, mu[i] / scale);
	}
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

	double *v = 0, *w = 0;
	uchar *im1 = new uchar[W*H];
	for(int i = 0; i < 1; i++) {
		file_im.read((char*) im1, W*H);
		file_lab.read(lab, 1);
		cout << (int) *lab << " " << wasserstein(im0, im1, W, H, 1.5, 10, v, w) << endl;
	}
	int N = 4;
	char name[100];
	for(int i = 0; i <= N; i++) {
		double j = (double) i / (double) N;
		uchar* bar = barycenter({im0, im1}, {j, 1-j}, W, H, 1.1, 7);
		sprintf(name, "bar%d.png", i);
		stbi_write_png(name, W, H, 1, bar, 0);
		delete[] bar;
	}

	delete[] im0;
	delete[] im1;
	delete[] v;
	delete[] w;

	int C;
	vector<uchar*> ims = {stbi_load("circle.png", &W, &H, &C, 1), stbi_load("circle_2.png", &W, &H, &C, 1),
						stbi_load("spiral.png", &W, &H, &C, 1), stbi_load("star.png", &W, &H, &C, 1)};
	for(uchar* im : ims)
		for(int i = 0; i < W*H; i++)
			im[i] = 255 - im[i];
	uchar *big = new uchar[(N+1)*W*(N+1)*H];
	for(int xi = 0; xi <= N; xi ++) {
		for(int yi = 0; yi <= N; yi ++) {
			double x = (double) xi / (double) N;
			double y = (double) yi / (double) N;
			uchar *bar = barycenter(ims, {x*y, (1-x)*y, x*(1-y), (1-x)*(1-y)}, W, H, 1.3, 7);
			uchar ma = 0;
			for(int i = 0; i < W*H; i++)
				if(bar[i] < 200)
					ma = max(ma, bar[i]);
			ma = (double) ma * 0.68;
			for(int i = 0; i < W*H; i++)
				if(bar[i] < ma)
					bar[i] = 0;
			for(int i = 0; i < W; i++)
				for(int j = 0; j < H; j++)
					big[W*xi + i + W*(N+1) * (H*yi + j)] = bar[i + W*j];
			delete[] bar;
		}
	}
	stbi_write_png("big.png", W*(N+1), H*(N+1), 1, big, 0);
	for(uchar* im : ims) free(im);
	delete[] big;

	im0 = stbi_load("eight.png", &W, &H, &C, 3);
	im1 = stbi_load("nine.png", &W, &H, &C, 3);
	ims.clear();
	for(int c = 0; c < 3; c++) {
		uchar *im = new uchar[W*H];
		for(int i = 0; i < W*H; i++)
			im[i] = max(0.0, 255.0 - 1.28*im0[3*i+c]);
		ims.push_back(im);
		im = new uchar[W*H];
		for(int i = 0; i < W*H; i++)
			im[i] = max(0.0, 255.0 - 1.28*im1[3*i+c]);
		ims.push_back(im);
	}
	N = 14;
	for(int i = 0; i <= N; i++) {
		double j = (double) i / (double) N;
		uchar* res = new uchar[3*W*H];
		for(int c = 0; c < 3; c++) {
			uchar* bar = barycenter({ims[2*c], ims[2*c+1]}, {j, 1-j}, W, H, 1.45, 22);
			for(int j = 0; j < W*H; j++)
				res[3*j+c] = (255.0 - bar[j]) / 1.28;
			delete[] bar;
		}
		sprintf(name, "bar%d.png", i);
		stbi_write_png(name, W, H, 3, res, 0);
		delete[] res;
	}
	free(im0);
	free(im1);
	for(uchar* im : ims) delete[] im;

	return 0;
}