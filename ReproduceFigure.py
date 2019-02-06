# ~~~MODULES~~~ #
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import tensorflow as tf


# ~~~PARAMETERS~~~ #
plotHeight = 40  # y-axis plot height
modelName = 'HYnetBt'  # tensorflow model weights
parameters = [1, 1, 1, 1, 1, 0]  # predict a, T, d, K, and e
plotSelect = True  # toggle plotting of select and all experiments


# ~~~IMPORT DATA~~~ #
# Experimental flattened HYSCORE images for validation
spectraE = pd.read_csv('HYnetBt/Data/spectra.csv', header=None).values

# Reshape HYSCORE spectra as 3D matrices
pixelWidth = int(np.sqrt(len(spectraE[0])))
spectraE = np.reshape(spectraE, (len(spectraE), pixelWidth, pixelWidth))

# Experiment parameters
paramE = pd.read_csv('HYnetBt/Data/params.csv').values
paramE[:, -2:] /= 1000

# Experimental one-hot encoded classifications for validation
classEa = pd.read_csv('HYnetBt/Data/a.csv', header=None).values
classET = pd.read_csv('HYnetBt/Data/T.csv', header=None).values
classEd = pd.read_csv('HYnetBt/Data/d.csv', header=None).values
classEK = pd.read_csv('HYnetBt/Data/K.csv', header=None).values
classEe = pd.read_csv('HYnetBt/Data/e.csv', header=None).values
classEB = pd.read_csv('HYnetBt/Data/B.csv', header=None).values

# Number of classifications
classes_a = len(classEa[0])
classes_T = len(classET[0])
classes_d = len(classEd[0])
classes_K = len(classEK[0])
classes_e = len(classEe[0])
classes_B = len(classEB[0])


# ~~~ARCHITECTURE~~~ #
# Shared convolutional layers
cvNet1 = [3, 16]
cvNet2 = [3, 32]
cvNet3 = [3, 64]

# Shared fully connected layers
fcNet1 = 256

# Classification layers for a, T, d, K, e, and B
clNet1 = [64, 64, 64, 64, 64, 64]


# ~~~FUNCTIONS~~~ #
# Randomize weights
def weight_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)


# Initiate bias
def bias_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)


# Convolution with bias and ReLU
def conv_2d(h, w, b):
    convolution = tf.nn.conv2d(h, w, strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(convolution + b)


# Gaussian function
def gauss(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2/(2 * sigma**2))


# ~~~NEURAL NETWORK~~~ #
# Number of HYSCORE data points and output classes
numPixels = pixelWidth ** 2

# Input nodes
x_HY = tf.placeholder(tf.float32, [None, pixelWidth, pixelWidth])
x_param = tf.placeholder(tf.float32, [None, 2])
x_image = tf.reshape(x_HY, [-1, pixelWidth, pixelWidth, 1])
ya = tf.placeholder(tf.float32, [None, classes_a])
yT = tf.placeholder(tf.float32, [None, classes_T])
yd = tf.placeholder(tf.float32, [None, classes_d])
yK = tf.placeholder(tf.float32, [None, classes_K])
ye = tf.placeholder(tf.float32, [None, classes_e])
yB = tf.placeholder(tf.float32, [None, classes_B])
keep_prob = tf.placeholder(tf.float32)

# Magnetic field and tau tensors
x_field = tf.reshape(x_param[:, 0], [-1, 1, 1, 1])
x_field = tf.tile(x_field, [1, pixelWidth, pixelWidth, 1])
x_tau = tf.reshape(x_param[:, 0], [-1, 1, 1, 1])
x_tau = tf.tile(x_tau, [1, pixelWidth, pixelWidth, 1])

# Concatenate experimental settings
x_image = tf.concat([x_image, x_field], axis=-1)
x_image = tf.concat([x_image, x_tau], axis=-1)

# Convolutional layer 1
W_conv1 = weight_variable([cvNet1[0], cvNet1[0], 3, cvNet1[1]])
b_conv1 = bias_variable([cvNet1[1]])
h_conv1 = conv_2d(x_image, W_conv1, b_conv1)

# Convolutional layer 2
W_conv2 = weight_variable([cvNet2[0], cvNet2[0], cvNet1[1], cvNet2[1]])
b_conv2 = bias_variable([cvNet2[1]])
h_conv2 = conv_2d(h_conv1, W_conv2, b_conv2)

# Convolutional layer 3
W_conv3 = weight_variable([cvNet3[0], cvNet3[0], cvNet2[1], cvNet3[1]])
b_conv3 = bias_variable([cvNet3[1]])
h_conv3 = conv_2d(h_conv2, W_conv3, b_conv3)

# Flatten
h_convF = tf.reshape(h_conv3, [-1, numPixels * cvNet3[1] // 4**3])

# Fully Connected Layer 1
W_fc1 = weight_variable([h_convF.get_shape().as_list()[1], fcNet1])
b_fc1 = bias_variable([fcNet1])
h_fc1 = tf.nn.relu(tf.matmul(h_convF, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Classification layers
h_fcF = h_fc1_drop
W_a1 = weight_variable([h_fcF.get_shape().as_list()[1], clNet1[0]])
W_T1 = weight_variable([h_fcF.get_shape().as_list()[1], clNet1[1]])
W_d1 = weight_variable([h_fcF.get_shape().as_list()[1], clNet1[2]])
W_K1 = weight_variable([h_fcF.get_shape().as_list()[1], clNet1[3]])
W_e1 = weight_variable([h_fcF.get_shape().as_list()[1], clNet1[4]])
W_B1 = weight_variable([h_fcF.get_shape().as_list()[1], clNet1[5]])
b_a1 = bias_variable([clNet1[0]])
b_T1 = bias_variable([clNet1[1]])
b_d1 = bias_variable([clNet1[2]])
b_K1 = bias_variable([clNet1[3]])
b_e1 = bias_variable([clNet1[4]])
b_B1 = bias_variable([clNet1[5]])
h_a1 = tf.nn.relu(tf.matmul(h_fcF, W_a1) + b_a1)
h_T1 = tf.nn.relu(tf.matmul(h_fcF, W_T1) + b_T1)
h_d1 = tf.nn.relu(tf.matmul(h_fcF, W_d1) + b_d1)
h_K1 = tf.nn.relu(tf.matmul(h_fcF, W_K1) + b_K1)
h_e1 = tf.nn.relu(tf.matmul(h_fcF, W_e1) + b_e1)
h_B1 = tf.nn.relu(tf.matmul(h_fcF, W_B1) + b_B1)
h_a1_drop = tf.nn.dropout(h_a1, keep_prob)
h_T1_drop = tf.nn.dropout(h_T1, keep_prob)
h_d1_drop = tf.nn.dropout(h_d1, keep_prob)
h_K1_drop = tf.nn.dropout(h_K1, keep_prob)
h_e1_drop = tf.nn.dropout(h_e1, keep_prob)
h_B1_drop = tf.nn.dropout(h_B1, keep_prob)

# Output layer - a
W_aF = weight_variable([clNet1[0], classes_a])
b_aF = bias_variable([classes_a])
y_a = tf.nn.softplus(tf.matmul(h_a1_drop, W_aF) + b_aF)

# Output layer - T
W_TF = weight_variable([clNet1[1], classes_T])
b_TF = bias_variable([classes_T])
y_T = tf.nn.softplus(tf.matmul(h_T1_drop, W_TF) + b_TF)

# Output layer - d
W_dF = weight_variable([clNet1[2], classes_d])
b_dF = bias_variable([classes_d])
y_d = tf.nn.softplus(tf.matmul(h_d1_drop, W_dF) + b_dF)

# Output layer - K
W_KF = weight_variable([clNet1[3], classes_K])
b_KF = bias_variable([classes_K])
y_K = tf.nn.softplus(tf.matmul(h_K1_drop, W_KF) + b_KF)

# Output layer - e
W_eF = weight_variable([clNet1[4], classes_e])
b_eF = bias_variable([classes_e])
y_e = tf.nn.softplus(tf.matmul(h_e1_drop, W_eF) + b_eF)

# Output layer - B
W_BF = weight_variable([clNet1[5], classes_B])
b_BF = bias_variable([classes_B])
y_B = tf.nn.softplus(tf.matmul(h_B1_drop, W_BF) + b_BF)


# ~~~MODEL EVALUATION~~~ #
# Predict coupling constants
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Load pretrained weights
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(modelName))

    # Set up figure
    if plotSelect:
        experiments = [0, 4, 8, 10, 12, 13, 14, 15, 19]
    else:
        experiments = list(range(len(spectraE)))
    fig, ax = plt.subplots(nrows=len(experiments), ncols=5)
    roman = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix']
    col = 0

    # Plot function
    def exp_plot(param, exp, row, column):
        if param == -2:
            plot_exp = (paramE[exp][param] - 0.3) * 20
        else:
            plot_exp = paramE[exp][param]
        p0 = [np.max(expPred[0]), 0.1 * np.argmax(expPred[0]), 0.1]
        xRange = np.arange(0.1, round(0.1 * (len(expPred[0]) + 1), 1), 0.1)
        try:
            pOpt, pCov = curve_fit(gauss, xRange - 0.1/2, expPred[0], p0,
                                   bounds=(-0.001, [np.max(expPred[0]), 10.0, 10.0]))
        except ValueError:
            print('ValueError')
            pOpt = np.array([1, 0, 0])
        except RuntimeError:
            print('RuntimeError')
            pOpt = np.array([1, 0, 0])
        except:
            print('OptimizeWarning')
            pOpt = np.array([1, 0, 0])
        expReal = gauss(paramE[exp][param], *pOpt)
        condition = param != 3 and not (param == 5 and (row not in {0, 2, 3}))
        if paramE[exp][param] != 0.01 and (condition or not plotSelect):
            ax[row, column].plot([plot_exp, plot_exp],
                                 [0, expReal], linewidth=3)
            ax[row, column].scatter(plot_exp, expReal + 3, marker='v')
        ax[row, column].plot(xRange - 0.1 / 2, expPred[0], color='k')
        ax[row, column].plot(xRange - 0.1 / 2, gauss(xRange - 0.1 / 2, *pOpt),
                             'r--', linewidth=2)
        ax[row, column].set_xlim([0, xRange[-1]])
        ax[row, column].set_ylim([0, plotHeight])
        ax[row, column].set_xticks([])
        ax[row, column].set_yticks([])
        return pOpt

    # Preallocate fitting parameter matrices
    height = np.zeros((len(experiments), len(parameters)))
    center = np.zeros((len(experiments), len(parameters)))
    sigma = np.zeros((len(experiments), len(parameters)))

    # Plot probability distributions - a
    for (i, j) in enumerate(experiments):
        expPred = np.round(y_a.eval(feed_dict={
            x_HY: spectraE[j:j + 1],
            x_param: paramE[j:j + 1, -2:],
            ya: classEa[j:j + 1],
            keep_prob: 1.0}), decimals=1)
        height[i, col], center[i, col], sigma[i, col] = exp_plot(0, j, i, col)
        if plotSelect:
            ax[i, col].set_ylabel('(' + roman[i] + ')', labelpad=25, rotation=0,
                                  fontsize=15, weight='bold', style='italic', y=0.35)
    ax[0, col].set_title(r'$a$', fontsize=30, y=1.1)
    ax[-1, col].set_xticks([0, 2, 4, 6, 8])
    ax[-1, col].set_xticklabels(['0', '2', '4', '6', '8'],
                                fontsize=12, weight='bold')
    ax[-1, col].set_xlabel('(MHz)', fontsize=12, weight='bold')
    col += 1

    # Plot probability distributions - T
    for (i, j) in enumerate(experiments):
        expPred = np.round(y_T.eval(feed_dict={
            x_HY: spectraE[j:j + 1],
            x_param: paramE[j:j + 1, -2:],
            yT: classET[j:j + 1],
            keep_prob: 1.0}), decimals=1)
        height[i, col], center[i, col], sigma[i, col] = exp_plot(2, j, i, col)
    ax[0, col].set_title(r'$T$', fontsize=30, y=1.1)
    ax[-1, col].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax[-1, col].set_xticklabels(['0', '', '0.5', '', '1'],
                                fontsize=12, weight='bold')
    ax[-1, col].set_xlabel('(MHz)', fontsize=12, weight='bold')
    col += 1

    # Plot probability distributions - d
    for (i, j) in enumerate(experiments):
        expPred = np.round(y_d.eval(feed_dict={
            x_HY: spectraE[j:j + 1],
            x_param: paramE[j:j + 1, -2:],
            yd: classEd[j:j + 1],
            keep_prob: 1.0}), decimals=1)
        height[i, col], center[i, col], sigma[i, col] = exp_plot(3, j, i, col)
    ax[0, col].set_title(r'$\delta$', fontsize=30, y=1.1)
    ax[-1, col].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax[-1, col].set_xticklabels(['0', '', '0.5', '', '1'],
                                fontsize=12, weight='bold')
    col += 1

    # Plot probability distributions - K
    for (i, j) in enumerate(experiments):
        expPred = np.round(y_K.eval(feed_dict={
            x_HY: spectraE[j:j + 1],
            x_param: paramE[j:j + 1, -2:],
            yK: classEK[j:j + 1],
            keep_prob: 1.0}), decimals=1)
        height[i, col], center[i, col], sigma[i, col] = exp_plot(4, j, i, col)
    ax[0, col].set_title(r'$K$', fontsize=30, y=1.1)
    ax[-1, col].set_xticks([0, 1.0, 2.0, 3.0, 4.0, 5.0])
    ax[-1, col].set_xticklabels(['0', '1', '2', '3', '4', '5'],
                                fontsize=12, weight='bold')
    ax[-1, col].set_xlabel('(MHz)', fontsize=12, weight='bold')
    col += 1

    # Plot probability distributions - e
    for (i, j) in enumerate(experiments):
        expPred = np.round(y_e.eval(feed_dict={
            x_HY: spectraE[j:j + 1],
            x_param: paramE[j:j + 1, -2:],
            ye: classEe[j:j + 1],
            keep_prob: 1.0}), decimals=1)
        height[i, col], center[i, col], sigma[i, col] = exp_plot(5, j, i, col)
    ax[0, col].set_title(r'$\eta$', fontsize=30, y=1.1)
    ax[-1, col].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax[-1, col].set_xticklabels(['0', '', '0.5', '', '1'],
                                fontsize=12, weight='bold')
    col += 1

    # Save Gaussian fit parameters
    np.savetxt(modelName + '/' + modelName + '_height',
               height, fmt='%8.3f', delimiter=',')
    np.savetxt(modelName + '/' + modelName + '_center',
               center, fmt='%8.3f', delimiter=',')
    np.savetxt(modelName + '/' + modelName + '_sigma',
               sigma, fmt='%8.3f', delimiter=',')

# Average deviation from literature values
literature = paramE[:, [0, 2, 3, 4, 5, -2]]
literature[:, -1] = (literature[:, -1] - 0.3) * 20
averages = []
for i in range(6):
    deviations = [abs(x - y) for x, y in
                  zip(center[:, i], literature[:, i]) if y != 0.01]
    if deviations:
        averages.append(sum(deviations) / len(deviations))
print('Average deviations')
print('a = %0.2f MHz' % averages[0])
print('T = %0.2f MHz' % averages[1])
print('K = %0.2f MHz' % averages[2])
print('e = %0.2f' % averages[3])

# Finalize plots
plt.tight_layout(pad=0, w_pad=1, h_pad=0)
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
plt.show()