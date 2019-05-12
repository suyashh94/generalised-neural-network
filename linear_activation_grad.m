function da = linear_activation_grad(z)
% calculating gradient because of linear activation
da = z;
da(da >= 0) = 1;
da(da < 0) = 0;
