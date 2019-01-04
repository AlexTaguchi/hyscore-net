# hyscore-net
Convolutional neural network for analyzing single-nuclear <sup>14</sup>N HYSCORE spectra

### Installation Instructions
[1] Purchase, download, and install Matlab (https://www.mathworks.com/products/matlab.html)

[2] Download EasySpin (http://easyspin.org/download.html) and follow installation instructions

[3] Download and install the Python 3 version of Anaconda (https://www.anaconda.com/download/)

[4] Open a terminal or command prompt and install TensorFlow by typing "pip install tensorflow"

### Installation notes
- Matlab is required to access EasySpin's eprload function to import HYSCORE spectra. Octave (the open source version of Matlab) will not work as EasySpin's functions were written in Matlab's proprietary format.
- While not required, Anaconda is recommended as the most convinient way to install all of the required Python package dependencies (matplotlib, numpy, pandas, and scipy).
- TensorFlow (created by Google) is the machine learning package used for training and evaluating the convolutional neural network (CNN).

### Usage
[1] Click the green "Clone or Download" button and "Download ZIP". Extract in desired location on your computer.

[2] Open a terminal or command prompt in this extracted directory and type "python ReproduceFigure.py" to run the CNN with the example HYSCORE spectra to reproduce the figure in the paper.

[3] Run a single-nuclear <sup>14</sup>N HYSCORE spectrum by replacing mq_52A.par and mq_52A.spc in the Spectra folder with a phased HYSCORE time-domain pattern of your own (or leave the files alone to analyze the example spectrum). Then open HYnetBt.m in Matlab, adjust the filename, field, and tau values on lines 7-9 accordingly, and run the script in Matlab.

### Publication
<i>in review</i>
