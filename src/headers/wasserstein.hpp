#ifndef WASSERSTEIN_DEF
#define WASSERTEIN_DEF

#include <vector>

#define uchar unsigned char

double wasserstein(uchar *im0, uchar *im1, int W, int H, double sigma, int num_it,
				double *av, double *aw);

uchar* barycenter(std::vector<uchar*> ims, std::vector<double> alpha, int W, int H, double sigma, int num_it);

#undef uchar

#endif