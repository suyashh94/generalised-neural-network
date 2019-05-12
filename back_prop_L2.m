function dW_L2 = back_prop_L2(dW,W,lambda)
m = size(W{1},2);
num_layers = length(W);
for i = 1:num_layers
    dW_L2{i} = 1/m*(lambda*W{i});
    dW_L2{i} = dW{i} + dW_L2{i};
end