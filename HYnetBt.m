% ~~~~Predict Hyperfine and Nuclear Quadrupole~~~~~ %
% ~~~~~Tensors from Single-Nuclear 14N HYSCORE~~~~~ %
% ~~~Spectrum with Convolutional Neural Networks~~~ %


% ===============PARAMETERS TO CHANGE============== %
filename = 'mq_52A.spc'; % HYSCORE time-domain pattern
field = 344.4; % magnetic field (mT)
tau = 136; % delay between the 1st and 2nd pulses (ns)
% ===========DO NOT CHANGE ANYTHING BELOW=========== %



% Preprocess time-domain pattern into square format
[xy,sp] = Preprocess(filename);
% contour(xy,xy,reshape(sp,128,128),30,'LineWidth',1.5)

% Export spectrum and parameters for neural network
folder = 'Spectra/Preprocessed/';
params = array2table(round([field tau],2),...
    'VariableNames', {'field','tau'});
writetable(params,strcat(folder,'params.csv'))
csvwrite(strcat(folder,'spectra.csv'),round(sp,3))

% Predict coupling constants with TensorFlow
system('python HYnetBt/HYnetBt.py');