clear all

% Make a training set 
load('C:\Users\harla\Desktop\Matlab scripts\Datasets\mnist.mat');
x_train = [];
for i = 1:10
    temp = ['train' num2str(i-1)];
    x_train = [x_train ; eval(temp)];
end
x_train = x_train';

total_sample = 0;
y_train = zeros(10,size(x_train,2));
for i = 1:10
    temp = ['train' num2str(i-1)];
    y_train(i,total_sample+1:total_sample + size(eval(temp),1)) = 1 ;
    total_sample = total_sample + size(eval(temp),1);
end

% Make test set
x_test = [];
for i = 1:10
    temp = ['test' num2str(i-1)];
    x_test = [x_test ; eval(temp)];
end
x_test = x_test';

total_sample = 0;
y_test = zeros(10,size(x_test,2));
for i = 1:10
    temp = ['test' num2str(i-1)];
    y_test(i,total_sample+1:total_sample + size(eval(temp),1)) = 1 ;
    total_sample = total_sample + size(eval(temp),1);
end

save('training_data.mat','x_train','y_train');
save('test_data.mat','x_test','y_test');