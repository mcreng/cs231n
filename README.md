## Stanford CS231n (HKUST COMP4901J) Assignment Repo

### Introduction
This repository stores the assignments I have done in HKUST COMP4901J - Deep Learning for Computer Vision, which is a course adapted from [Stanford CS231n - Convolutional Neural Networks for Visual Recoginition](http://cs231n.github.io/).

There are four sets of assignments in total, in which the first three are adapted from Stanford, and the fourth is created by HKUST.

### Assignment 1
* `knn.ipynb`: k-NN with Cross Validation using pure Numpy.
* `svm.ipynb`: SVM classifier on CIFAR-10 using pure Numpy.
* `softmax.ipynb`: Softmax classifier on CIFAR-10 using pure Numpy.
* `two_layer_net.ipynb`: NN with 2 FC layers on CIFAR-10 using pure Numpy.
* `feature.ipynb`: Training classifiers on CIFAR-10 with image features (Histogram of Gradients, Color Histrogram) using pure Numpy.
* Bonus 1 (in `feature.ipynb`): Further training networks with different combinations of features.
* Bonus 2 (in `feature.ipynb`): Using a NN with 2 FC layers on MNIST using pure Numpy.

### Assignment 2
* `FullyConnectedNets.ipynb`: Neural Network with n FC layers and ReLU activations on CIFAR-10 with various optimization algorithms (SGD, momentum, RMSProp and Adam) using pure Numpy.
* `Dropout.ipynb`: Dropout layer implementation in pure Numpy.
* `BatchNormalization.ipynb`: Batch Normalization layer implementation in pure Numpy.
* `ConvolutionalNetworks.ipynb`: CNN implementation in pure Numpy.
* `Tensorflow.ipynb`: Training networks on CIFAR-10 using tensorflow. Achieved 80% on test set.
* Bonus 1 (in `FullyConnectedNets.ipynb`): Early stopping in model training.
* Bonus 2 (in `ConvolutionalNetworks.ipynb`): Average Pooling layer implementation in pure Numpy.
* Bonus 3 (in `Tensorflow.ipynb`): CNN kernel visualizations.

### Assignment 3
* `RNN_Captioning.ipynb`: Simple RNN implementation using pure Numpy on Coco dataset for image captioning.
* `LSTM_Captioning.ipynb`: LSTM implementation using pure Numpy on Coco dataset for image captioning.
* `NetworkVisualization-Tensorflow.ipynb`: Saliency Map, Fooling Images and Class Visualization on tensorflow.
* `StyleTransfer-Tensorflow.ipynb`: Style Transfer network implementation on tensorflow.
* `GANs-Tensorflow.ipynb`: Ordinary GAN, LSGAN, DCGAN implementations on tensorflow.
* Bonus 1 (in `GANs-Tensorflow.ipynb`):  WGAN-GP implementation on tensorflow.
* Bonus 2 (in `Bonus-CGAN.ipynb`): CGAN implementation on tensorflow.
* Bonus 3 (in `Bonus-InfoGAN.ipynb`): InfoGAN implementation on tensorflow.
* Bonus 4 (in `Bonus-SemiSupervisedLearning.ipynb`): GAN for semi-supervised learning on MNIST. Result is not the greatest.
  
### Assignment 4
* `Q_learning_basic.ipynb`: Q-table and Q-learning on gym using tensorflow.
* `DQN_WorldNavigate.ipynb`: DQN, Double/Dueling DQN on gym using tensorflow.
* `PG_CartPole.ipynb`: Policy Gradient on gym using tensorflow.
* `Model_Policy_Network.ipynb`: Model-based Policy Gradient on gym using tensorflow.
* Bonus 1 (in `Bonus-DQRN.ipynb`): DQRN implementation on gym using tensorflow.
* Bonus 2 (in `Bonus-DQN_WorldNavigate-HyperparameterTuning.ipynb`): DQN with tuned hyperparameters.
* Bonus 3 (in `Bonus-Model_Policy_Network-HyperparameterTuning.ipynb`): Model-based Policy Gradient network with tuned hyperparamenters.
* Bonus 4 (in `Bonus-DQN_WorldNavigate-Breakout.ipynb`): DQN on Atari Breakout. Result is not the best.
* Bonus 5 (in `Bonus-Curiosity-MountainCar.ipynb`): Curiosity driven RL implementation using tensorflow.