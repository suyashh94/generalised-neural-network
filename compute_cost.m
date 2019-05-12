function cost = compute_cost(y_train,A)
m = size(y_train,2); % No. of training samples
aL = A{end};
aL(aL == 0) = 10^-10;
aL(aL == 1) = 0.99;
y = y_train;
cost = y.*log(aL) + (1-y).*log(1-aL);
cost = -1/m*(sum(sum(cost)));
