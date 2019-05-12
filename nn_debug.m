clear all;
%% Load training data
load('C:\Users\harla\Desktop\Matlab scripts\Datasets\MNIST Fashion\training_set.mat');
%% Normalize training data - here just diving by 255
% x_train = im2double(x_train);
x_train = x_train(:,1:20000);
y_train = y_train(:,1:20000);
t = shuffle(1:20000);
x_train = x_train(:,t);
y_train = y_train(:,t);
% x_train = x_train/255;
learning_rate = 0.9;  % Increase it
%% Number of layers
num_layers = input('Enter number of layers excluding input layer ');
layer_dims = zeros(num_layers+1,1);
layer_dims(1) = size(x_train,1);
layer_dims(end) = size(y_train,1);
for i = 2:length(layer_dims)-1
    layer_dims(i) = input(['Enter number of units in layer ' num2str(i)]);
end
%% Initialize weights of all hidden layers
[W, b] = initialize_parameters(layer_dims);
W_intial = W;
b_initial = b;
%% Defining activations for every layer
% Last layer has activation sigmoid
% All layers before it has activation relu
for i = 1:num_layers
    if i ~= num_layers
        if mod(i,2) == 0
            activation{i} = 'relu';
        else
            activation{i} = 'sigmoid';
        end
    else
        activation{i} = 'sigmoid';
    end
end

%% Start the training loop
for iter = 1 : 2000
    iter
    %% Forward propagation of the network
    [A,Z] = forward_propagation(x_train,W,b,activation);
    %% Computing cost from last layer activation
    cost(iter) = compute_cost(y_train,A);
    %% Backward propagation
    [dW ,db] = backward_propagation(x_train,y_train,A,Z,W,activation);
    %% Update Parameters - normal gradient descent
    for i = 1:num_layers
        W{i} = W{i} - learning_rate * dW{i};
        b{i} = b{i} - learning_rate * db{i};
    end
    
    %% end the training loop
end


%% plot cost
figure;
plot(cost)

%% plot A
figure;
imagesc(A{end})

