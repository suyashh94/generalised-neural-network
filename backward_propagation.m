function [dW ,db] = backward_propagation(x_train,y_train,A,Z,W,activation)
m = size(y_train,2);
num_layers = length(A);
dZ = cell(1,num_layers);
if strcmp(activation{end},'sigmoid')
    dZ{end} = A{end} - y_train;
end

for i = num_layers : -1 : 2
    if strcmp(activation{i-1},'relu')
        dZ{i-1} = (W{i}' * dZ{i}) .* linear_activation_grad(Z{i-1});
    elseif strcmp(activation{i-1},'sigmoid')
        dg = sigmoid_activation_grad(Z{i-1});
        dg(isnan(dg)) = 0;
        dZ{i-1} = (W{i}' * dZ{i}) .* dg;
    end
end

for i = 1:num_layers
    if i == 1
        dW{i} = 1/m * (dZ{i}*x_train');
    else
        dW{i} = 1/m * (dZ{i}*A{i-1}');
    end
    
    db{i} = 1/m * sum(dZ{i},2);
end