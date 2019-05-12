function [gamma, beta] = initialize_gamma_and_beta(layer_dims)
% This functions returns weights for every layer 
% W{1xlayer_dims-1} - each cell having weights of every (l-1) layer
% b{1xlayer_dims-1} - each cell having randomly initialized value of b. 
% Initializing b to zero
% Input is layer_dims - a vector having number of units in each layer

for i = 1:length(layer_dims)-1
    gamma{i} = ones(layer_dims(i+1) , 1) ;
    beta{i} = zeros(layer_dims(i+1) , 1);
end 

