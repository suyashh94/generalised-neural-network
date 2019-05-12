function a = sigmoid_activation(z)
% This does sigmoid activation on WX = Z 
a = 1./(1+exp(-z));