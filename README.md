# Bezier-EGAN

Bezier GAN's application to UIUC airfoil database realized within the framework of Entropic GAN.

## Environment

* Python 3.7
* PyTorch 1.6.0
* Tensorboard 2.2.1
* Numpy 1.19.1
* Scipy 1.5.2

## Scripts

* **train**: _Training algorithm where all elements developed are assembled._
* **models**
  * **layers**: _Elementary PyTorch modules to be embedded in cmpnts._
    * Bezier layer generating data points.
    * Regular layer combos for components in cmpnts.
  * **cmpnts**: _General PyTorch neural network components for advanced applications._
    * Basic components such as MLP, Conv1d front end etc.
    * Generators for a variety of applications.
    * Discriminators for a variety of applications.
  * **gans**: _Various GAN containers built on top of each other._
    * GAN: Trained with JS divergence.
    * EGAN: Child of GAN trained with entropic dual loss.
    * InfoEGAN: Child of EGAN trained with additional mutual information maximization.
    * BezierEGAN: Child of InfoGAN trained with additional bezier curve regularization.
  * **utils**: _Miscellaneous tools_
* **utils**
  * **dataloader**: _Data related tools for GAN's training process._
    * Dynamic UIUC dataset that can generate samples with given parameters.
    * Noise generator producing a given batch of uniform and normal noise combination.
  * **interpolation**: _Interpolation algorithm._
  * **metrics**:
    * MMD
    * Consistency