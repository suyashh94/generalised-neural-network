
function [W,b] = update_parameters(W,b,learning_rate,dW,db)

num_layers = length(W);

for i = 1:num_layers
        W{i} = W{i} - learning_rate * dW{i};
        b{i} = b{i} - learning_rate * db{i};
end