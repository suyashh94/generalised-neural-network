clear all 
%% Training set
x_train = loadMNISTImages('train-images-idx3-ubyte');
y_train = loadMNISTLabels('train-labels-idx1-ubyte');
y_train = y_train';
% temp = reshape(images,28,28,60000);
% 
% figure; 
% imagesc(labels')
% 
% figure;
% imagesc(temp(:,:,2));

save('C:\Users\harla\Desktop\Matlab scripts\Datasets\MNIST Fashion\training_set.mat',...
    'x_train','y_train');
clear temp
%% Test set
x_test = loadMNISTImages('t10k-images-idx3-ubyte');
y_test = loadMNISTLabels('t10k-labels-idx1-ubyte');
y_test = y_test';
temp = reshape(x_test,28,28,10000);

% figure; 
% imagesc(y_test')
% 
% figure;
% imagesc(temp(:,:,2));

save('C:\Users\harla\Desktop\Matlab scripts\Datasets\MNIST Fashion\test_set.mat',...
    'x_test','y_test');