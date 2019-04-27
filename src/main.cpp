#include <iostream>
#include <fstream>
#include "stb_image_write.h"
#include "stb_image.h"
#include "wasserstein.hpp"
#include "examples.hpp"
#include <cmath>
#include <cstring>

using namespace std;

#define N_TRAIN 500
#define N_TEST 1500

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

double euclidean(uchar* a, uchar* b, int W, int H) {
	long long dis = 0;
	#pragma omp parallel for reduction(+:dis)
	for(int k = 0; k < W*H; k++) dis += pow((int) a[k] - (int) b[k], 2);
	double dist = sqrt(dis);
	return dist / 100;
}

void compute_kernel(int N, ifstream &file_im, std::streampos pos_train,
					int W, int H, uchar *im0, uchar *im1,
					bool wass, double *v, double *w) {
	for(int i = 0; i < N; i++) {
		file_im.seekg(pos_train + std::streamoff(i*W*H));
		file_im.read((char*) im0, W*H);
		file_im.seekg(pos_train);
		for(int j = 0; j < N_TRAIN; j++) {
			file_im.read((char*) im1, W*H);
			double dist = wass ? wasserstein(im0, im1, W, H, 1.5, 6, v, w)
								: euclidean(im0, im1, W, H);
			cout << dist << " ";
		}
		cout << "\n";
	}
}

void print_err() {
	cerr << "Error: Need exactly one argument. It can be:" << endl;
	cerr << "\t- card: to compute interpolation of two cards" << endl;
	cerr << "\t- bar: to compute computes barycenters of four shapes" << endl;
	cerr << "\t- wass: to compute the wasserstein kernel" << endl;
	cerr << "\t- eucl: to compute the euclidean kernel" << endl;
}

int main(int argc, char* argv[]) {
	if(argc != 2) {
		print_err();
		return 1;
	}
	bool wass;
	if(std::strcmp(argv[1], "card") == 0) {
		cards();
		return 0;
	} else if(std::strcmp(argv[1], "bar") == 0) {
		bar_shapes();
		return 0;
	} else if(std::strcmp(argv[1], "wass") == 0) {
		wass = true;
	} else if(std::strcmp(argv[1], "eucl") == 0) {
		wass = false;
	} else {
		print_err();
		return 1;
	}

	ifstream file_im("ims/train_images");
	ifstream file_lab("ims/train_labels");
	if(file_im.fail() || file_lab.fail()) {
		cerr << "Error while loading ims/train_images and ims/train_labels" << endl;
		return 1;
	}

	read_int(file_im);
	int num_images = read_int(file_im);
	int W = read_int(file_im);
	int H = read_int(file_im);
	cerr << "number of images: " << num_images << endl;
	cerr << "Dimension: " << W << " x " << H << endl;

	read_int(file_lab);
	read_int(file_lab);

	// Read some images an labels
	char lab;
	uchar *im0 = new uchar[W*H];
	uchar *im1 = new uchar[W*H];
	for(int i = 0; i < 300; i++) {
		file_lab.read(&lab, 1);
		file_im.read((char*) im0, W*H);
	}
	std::streampos pos_train = file_im.tellg();

	// Read labs of training set
	for(int i = 0; i < N_TRAIN; i++) {
		file_lab.read(&lab, 1);
		cout << (int) lab << " ";
	}
	cout << "\n";

	// Compute the kernel on the training set
	double *v = 0, *w = 0;
	compute_kernel(N_TRAIN, file_im, pos_train, W, H, im0, im1, wass, v, w);

	// Read labs of testing set
	for(int i = 0; i < N_TEST; i++) {
		file_lab.read(&lab, 1);
		cout << (int) lab << " ";
	}
	cout << "\n";
	
	// Compute the kernel on the testing set
	compute_kernel(N_TEST, file_im, pos_train, W, H, im0, im1, wass, v, w);

	if(!v) delete[] v;
	if(!w) delete[] w;

	delete[] im0;
	delete[] im1;

	// int N = 4;
	// char name[100];
	// for(int i = 0; i <= N; i++) {
	// 	double j = (double) i / (double) N;
	// 	uchar* bar = barycenter({im0, im1}, {j, 1-j}, W, H, 1.1, 4);
	// 	sprintf(name, "res/bar%d.png", i);
	// 	stbi_write_png(name, W, H, 1, bar, 0);
	// 	delete[] bar;
	// }

	return 0;
}