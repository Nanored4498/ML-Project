#include "wasserstein.hpp"
#include "stb_image_write.h"
#include "stb_image.h"
#include <cstdlib>
#include <iostream>

typedef unsigned char uchar;

void cards() {
	int W, H, C;
	uchar *im0 = stbi_load("ims/eight.png", &W, &H, &C, 3);
	uchar *im1 = stbi_load("ims/nine.png", &W, &H, &C, 3);

	std::vector<uchar*> ims;
	for(int c = 0; c < 3; c++) {
		uchar *im = new uchar[W*H];
		for(int i = 0; i < W*H; i++)
			im[i] = std::max(0.0, 255.0 - 1.28*im0[3*i+c]);
		ims.push_back(im);
		im = new uchar[W*H];
		for(int i = 0; i < W*H; i++)
			im[i] = std::max(0.0, 255.0 - 1.28*im1[3*i+c]);
		ims.push_back(im);
	}

	int N = 14;
	char name[100];

	for(int i = 0; i <= N; i++) {
		double j = (double) i / (double) N;
		uchar* res = new uchar[3*W*H];
		for(int c = 0; c < 3; c++) {
			uchar* bar = barycenter({ims[2*c], ims[2*c+1]}, {j, 1-j}, W, H, 3.0, 15);
			for(int j = 0; j < W*H; j++)
				res[3*j+c] = (255.0 - bar[j]) / 1.28;
			delete[] bar;
		}
		sprintf(name, "res/bar%d.png", i);
		stbi_write_png(name, W, H, 3, res, 0);
		delete[] res;
	}

	free(im0);
	free(im1);
	for(uchar* im : ims) delete[] im;
}

void bar_shapes() {
	int N = 4;
	int W, H, C;
	std::vector<uchar*> ims = {stbi_load("ims/circle.png", &W, &H, &C, 1), stbi_load("ims/circle_2.png", &W, &H, &C, 1),
						stbi_load("ims/spiral.png", &W, &H, &C, 1), stbi_load("ims/star.png", &W, &H, &C, 1)};
	for(uchar* im : ims)
		for(int i = 0; i < W*H; i++)
			im[i] = 255 - im[i];
	uchar *big = new uchar[(N+1)*W*(N+1)*H];
	for(int xi = 0; xi <= N; xi ++) {
		for(int yi = 0; yi <= N; yi ++) {
			double x = (double) xi / (double) N;
			double y = (double) yi / (double) N;
			uchar *bar = barycenter(ims, {x*y, (1-x)*y, x*(1-y), (1-x)*(1-y)}, W, H, 2.8, 15);
			uchar ma = 0;
			for(int i = 0; i < W*H; i++)
				if(bar[i] < 200)
					ma = std::max(ma, bar[i]);
			ma = (double) ma * 0.6;
			for(int i = 0; i < W*H; i++)
				if(bar[i] < ma)
					bar[i] = 0;
			for(int i = 0; i < W; i++)
				for(int j = 0; j < H; j++)
					big[W*xi + i + W*(N+1) * (H*yi + j)] = bar[i + W*j];
			delete[] bar;
		}
	}
	stbi_write_png("res/big.png", W*(N+1), H*(N+1), 1, big, 0);
	for(uchar* im : ims) free(im);
	delete[] big;
}