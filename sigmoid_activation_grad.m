function da = sigmoid_activation_grad(z)
da = exp(-z)./((1+exp(-z)).^2);