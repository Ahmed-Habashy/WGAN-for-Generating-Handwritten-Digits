# WGAN-for-Generating-Handwritten-Digits
This code is an implementation of the Wasserstein GAN (WGAN) algorithm for generating handwritten digits. The WGAN algorithm was introduced as a way to address the problem of training instability in the original GAN algorithm. 
This code contains an implementation of a Wasserstein Generative Adversarial Network (WGAN) used to generate handwritten digits from the MNIST dataset. The WGAN is a variant of GAN that uses the Wasserstein distance metric instead of the traditional Jenson-Shannon divergence for training the generator and the critic networks.

The code is written in Python and uses the Keras deep learning library with the Tensorflow backend.

# Prerequisites
Python 3.x
NumPy
Keras
TensorFlow
Matplotlib
# Files
wgan_handwritten_digits.py: The main script containing the implementation of the WGAN model.
README.md: This file.
# Usage
To run the script, simply execute the following command:

python wgan_handwritten_digits.py

By default, the script will generate 25,000 new handwritten digits and save them in the 'generated_images' folder. The script also saves the generator and critic models after training for later use.

# Model Architecture
The WGAN consists of two networks: a generator network and a critic network.

The critic network is a convolutional neural network (CNN) that takes as input a 28x28 grayscale image and outputs a single scalar value. The architecture of the critic consists of two convolutional layers with 64 filters each, followed by two fully connected layers. The layers are activated using the LeakyReLU activation function and the weights are clipped to a hypercube to enforce the Lipschitz constraint. The critic is trained to maximize the Wasserstein distance between the real and generated images.

The generator network is a deep neural network that takes as input a vector of random noise and outputs a 28x28 grayscale image. The architecture of the generator consists of a fully connected layer followed by two convolutional transpose layers. The layers are activated using the LeakyReLU activation function and batch normalization is used to stabilize the training. The generator is trained to minimize the Wasserstein distance between the real and generated images.

The critic and generator networks are trained alternatively. In each iteration, the critic network is trained for a fixed number of iterations while the generator network is updated once. This process is repeated until convergence.

# Acknowledgments
This code is adapted from the WGAN implementation in the book "Generative Deep Learning" by David Foster.
