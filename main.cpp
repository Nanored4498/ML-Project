#include <iostream>
#include <fstream>
#include "stb_image_write.h"
#include "stb_image.h"
#include "wasserstein.hpp"
#include "examples.hpp"

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

int main() {
	bar_shapes();
	return 0;

	ifstream file_im("ims/train_images");
	ifstream file_lab("ims/train_labels");

	read_int(file_im);
	int num_images = read_int(file_im);
	cerr << "number of images: " << num_images << endl;
	int W = read_int(file_im);
	int H = read_int(file_im);

	read_int(file_lab);
	read_int(file_lab);

	char lab;
	uchar *im0 = new uchar[W*H];
	for(int i = 0; i < 300; i++) {
		file_lab.read(&lab, 1);
		file_im.read((char*) im0, W*H);
	}
	// int lab0 = lab;
	// cout << (int) lab << endl;
	// int nums[10];
	// double dists[10];
	// for(int i = 0; i < 10; i++)
	// 	nums[i] = 0, dists[i] = 0;

	int N_Train = 80;
	int N_Test = 50;
	double *v = 0, *w = 0;
	uchar *im1 = new uchar[W*H];
	std::streampos pos_train = file_im.tellg();
	for(int i = 0; i < N_Train; i++) {
		file_lab.read(&lab, 1);
		cout << (int) lab << " ";
	}
	cout << "\n";
	for(int i = 0; i < N_Train; i++) {
		file_im.seekg(pos_train + std::streamoff(i*W*H));
		file_im.read((char*) im0, W*H);
		file_im.seekg(pos_train);
		for(int j = 0; j < N_Train; j++) {
			file_im.read((char*) im1, W*H);
			double dist = wasserstein(im0, im1, W, H, 1.5, 6, v, w);
			cout << dist << " ";
		}
		cout << "\n";
	}
	for(int i = 0; i < N_Test; i++) {
		file_lab.read(&lab, 1);
		cout << (int) lab << " ";
	}
	cout << "\n";
	for(int i = 0; i < N_Test; i++) {
		file_im.seekg(pos_train + std::streamoff((N_Train+i)*W*H));
		file_im.read((char*) im0, W*H);
		file_im.seekg(pos_train);
		for(int j = 0; j < N_Train; j++) {
			file_im.read((char*) im1, W*H);
			double dist = wasserstein(im0, im1, W, H, 1.5, 6, v, w);
			cout << dist << " ";
		}
		cout << "\n";
	}

	
	// for(int i = 0; i < 200; i++) {
	// 	file_im.read((char*) im1, W*H);
	// 	file_lab.read(&lab, 1);
	// 	double dist = wasserstein(im0, im1, W, H, 1.5, 6, v, w);
	// 	cout << (int) lab << " " << dist << endl;
	// 	nums[(int) lab] ++;
	// 	dists[(int) lab] += dist;
	// 	delete[] v;
	// 	delete[] w;
	// }

	// cout << "----- " << lab0 << " ------\n"; 
	// for(int i = 0; i < 10; i++)
	// 	cout << i << " " << dists[i] / nums[i] << endl;

	// int N = 4;
	// char name[100];
	// for(int i = 0; i <= N; i++) {
	// 	double j = (double) i / (double) N;
	// 	uchar* bar = barycenter({im0, im1}, {j, 1-j}, W, H, 1.1, 4);
	// 	sprintf(name, "res/bar%d.png", i);
	// 	stbi_write_png(name, W, H, 1, bar, 0);
	// 	delete[] bar;
	// }

	delete[] im0;
	delete[] im1;

	//cards();

	return 0;
}