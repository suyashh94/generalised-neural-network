function [A,Z] = forward_propagation(x_train,W,b,activation)
% This does the forward propagation
% Input are x_train in n_feature X n_training_sample
% W,b and activation method for each layer - relu or sigmoid
% Returns A and Z for every layer
num_training_samples = size(x_train,2);
num_layers = length(W);
assert(length(activation) == num_layers);
assert(length(b) == num_layers);
A_prev = x_train;
for i = 1:num_layers
    Z{i} = W{i} * A_prev + b{i};
    if strcmp(activation{i},'relu')
        A{i} = linear_activation(Z{i});
        assert(size(A{i},2) == num_training_samples);
    elseif strcmp(activation{i},'sigmoid')
        A{i} = sigmoid_activation(Z{i});
        assert(size(A{i},2) == num_training_samples);
    end 
    A_prev = A{i};
end 