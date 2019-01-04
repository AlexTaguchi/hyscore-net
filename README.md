# hyscore-net
Convolutional neural network for analyzing single-nuclear 14N HYSCORE spectra

### Installation Instructions
[1] Purchase, download, and install Matlab (https://www.mathworks.com/products/matlab.html)

[2] Download EasySpin (http://easyspin.org/download.html) and follow installation instructions

[3] Download and install the Python 3 version of Anaconda (https://www.anaconda.com/download/)

[4] Open a terminal or command prompt and install TensorFlow by typing "pip install tensorflow"

### Installation notes
- Matlab is required to access EasySpin's eprload function to import HYSCORE spectra. Octave (the open source version of Matlab) will not work as EasySpin's functions were written Matlab's proprietary format.
- While not required, Anaconda is recommended as the most convinient way to install all of the required Python package dependencies (matplotlib, numpy, pandas, and scipy).
- TensorFlow (created by Google) is the machine learning package used for training and evaluating the convolutional neural networks.

### Usage
