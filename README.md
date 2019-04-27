# Convolutional Wasserstein Distance Implementation

You need to download the train image dataset of the MNIST database [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/). To do so, run the script [get_mnist.sh](get_mnist.sh) which will download and rename files correctly.

To compile the code you can use the makefile
```
make
```
Then you can generate the wasserstein kernel by typing:
```
./main wass > kernel.txt
```
Next you can try this kernel on a SVM in python with:
```
python3 learning.py < kernel.txt
```
