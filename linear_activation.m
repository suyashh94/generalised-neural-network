function a = linear_activation(z)
% This does RELU activation on WX = Z. 
a = z;
a(a<0) = 0;
