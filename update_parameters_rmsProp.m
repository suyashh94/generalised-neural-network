
function [W,b] = update_parameters_rmsProp(W,b,learning_rate,dW,db,S_dW,S_db)

num_layers = length(W);
eps = 10^-8;

for i = 1:num_layers
        W{i} = W{i} - learning_rate * (dW{i}./(sqrt(S_dW{i}) + eps));
        b{i} = b{i} - learning_rate * (db{i}./(sqrt(S_db{i}) + eps));
end