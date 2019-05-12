function [W, b] = initialize_parameters(layer_dims)
% This functions returns weights for every layer 
% W{1xlayer_dims-1} - each cell having weights of every (l-1) layer
% b{1xlayer_dims-1} - each cell having randomly initialized value of b. 
% Initializing b to zero
% Input is layer_dims - a vector having number of units in each layer

for i = 1:length(layer_dims)-1
    W{i} = rand(layer_dims(i+1) , layer_dims(i)) .* 0.01 ;
%     W{i} = W{i} - mean(mean(W{i}));
    n = layer_dims(i);
    W{i} = W{i}*sqrt(2/n);
    b{i} = zeros(layer_dims(i+1),1);
end 

