# fast_knn_gpu
This is a tensorflow implementation of the algorithm for fast and efficient k-nearest neighbour (KNN) search on GPU as presented by Lukasz Kaiser, Ofir Nachun, Aurko Roy, and Samy Bengio in their paper "[Learning To Remember Rare Events](https://openreview.net/pdf?id=SJTQLdqlg)". Their code can be found [here](https://github.com/tensorflow/models/tree/master/learning_to_remember_rare_events). 

In this quick example I compare the speed of the tensorflow KNN implementation with the [KNeighborsClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier) from scikit-learn for classifying MNIST. For only 100 test images the KNeighborsClassifier takes about 5.7 seconds while the tensorflow implementation takes 0.7 seconds. However, as the number of test images is increased the computation time for the KNeighborsClassifier increases roughly linearly, while the tensorflow knn compute time barely increases. I am guessing that most of the compute time for the tensorflow implementation is actually used to transfer data to the GPU, rather than do the actual calculation. When the number of test images is increased to 10,000 the KNeighborsClassifier takes ~560 seconds (this is an approximation based on linear scaling), while the tensorflow implementation takes only 0.18 seconds (actually measured). That is a speed increase of ~3000x!

Here is a terminal output sample for running the script to classify 1,000 test images:

```
Extracting /tmp/data/train-images-idx3-ubyte.gz
Extracting /tmp/data/train-labels-idx1-ubyte.gz
Extracting /tmp/data/t10k-images-idx3-ubyte.gz
Extracting /tmp/data/t10k-labels-idx1-ubyte.gz
Evaluating sklearn KNN
Accuracy: 95.7%
Took 56.5752000809 seconds
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: TITAN X (Pascal), pci bus id: 0000:03:00.0)
Evaluating tensorflow KNN
Accuracy: 96.6%
Took 0.0756950378418 seconds
```

## Dependencies
tensorflow-gpu 1.0.0  
scikit-learn 0.18.1 
