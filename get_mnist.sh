cd ims
wget yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
mv train-images-idx3-ubyte.gz train_images.gz
mv train-labels-idx1-ubyte.gz train_labels.gz
gunzip -v train_images.gz
gunzip -v train_labels.gz