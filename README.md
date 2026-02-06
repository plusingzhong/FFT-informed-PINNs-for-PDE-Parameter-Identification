# FFT-informed-PINNs-for-PDE-Parameter-Identification
This work introduces an advanced physics-informed neural network (PINN) framework to identify unknown parameters within governing partial differential equations (PDEs) from scarce data. This symbolic reasoning approach seamlessly integrates FFT-informed Fourier features to mitigate spectral bias with a Neural Tangent Kernel (NTK) based adaptive weighting scheme to balance the training dynamics. Through automatic differentiation, the known PDE structure is embedded as a soft constraint, guiding the network to a physically consistent solution while simultaneously inferring the unknown coefficients. The resulting framework shows high efficacy for model parameterization in practical applications where collecting large datasets is intractable.
This repository provides supplementary codes and data for the following work:
* FFT-informed Fourier feature enhance physics-informed neural networks for inverse parameter identification of partial differential equations from scarce data
Software dependencies
This work was implemented on Tensorflow 1.15 in Python. The list of main software dependencies is:
* Anaconda
* Python 3.7
* Tensorflow 1.15
* numpy
* scipy
* matplotlib
* pyDOE
Installation guide
We recommend installing Python via Anaconda and then installing the listed packages.
How to run our cases?
To try out our simulations, you can visit the?FHN and Wave?directory and run the corresponding Python file. Most required data is included in the directory. The expected outcome will show the quality of the system response prediction and the accuracy of the parameter identification. Specifically, the output includes error metrics, the identified parameter values, and loss convergence plots for diagnostics. 
