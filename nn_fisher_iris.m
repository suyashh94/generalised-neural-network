clear all;
%% Load training data
load fisheriris
%% Normalize training data - here just diving by 255
% x_train = im2double(x_train);
x_main = meas';
t = shuffle(1:size(x_main,2));
x_train = x_main(:,t(1:120));
x_train_mu = mean(x_train,2);
x_train_std = (std(x_train',1))';
x_train = (x_train - x_train_mu);
x_train = x_train ./ x_train_std;
y_main = species;
y_all_main = zeros(3,size(x_main,2));

for i = 1:length(y_main)
    if strcmp(y_main{i},'setosa')
    y_all_main(1,i) = 1;
    elseif strcmp(y_main{i},'versicolor')
        y_all_main(2,i) = 1;
    else
        y_all_main(3,i) = 1;
    end 
end
y_main = y_all_main;
y_train = y_main(:,t(1:120));

%% Test set 

x_test = x_main(:,t(121:150));
x_test = x_test - x_train_mu;
x_test = x_test ./ x_train_std;
y_test = y_main(:,t(121:150));



%% Constant terms
% x_train = x_train/255;
learning_rate = 0.05;  % Increase it
c = 2;
lambda = 0.0;
drop_out = 1;

clear y_train_main
%% Make batches of training set

batch_size = 60;
batch_num = size(x_train,2)/batch_size;
for k = 1:batch_num
    x_train_mini_batch{k} = x_train(:,(k-1)*batch_size + 1 : k*batch_size);
    y_train_mini_batch{k} = y_train(:,(k-1)*batch_size + 1 : k*batch_size);
end

%% Number of layers
num_layers = input('Enter number of layers excluding input layer ');
layer_dims = zeros(num_layers+1,1);
layer_dims(1) = size(x_train,1);
layer_dims(end) = size(y_train,1);
for i = 2:length(layer_dims)-1
    layer_dims(i) = input(['Enter number of units in layer ' num2str(i)]);
end
%% Initialize weights of all hidden layers
[W, b] = initialize_parameters(layer_dims);
W_intial = W;
b_initial = b;
%% Keep_prob value
keep_prob = 0.6;
keep_prob_init = 1;
%% Initialize V_dw and V_db for gradient descent with momentum
[V_dW, V_db] = initialize_parameters_with_zero(layer_dims);
%% check that size of W and V_dW are same for every layer. Same thing for b
for i = 1:num_layers
    [a1 , a2] = size(W{i});
    [c1 ,c2] = size(V_dW{i});
    assert(a1 == c1);
    assert(a2 == c2);
end

for i = 1:num_layers
    [a1 , a2] = size(b{i});
    [c1 ,c2] = size(V_db{i});
    assert(a1 == c1);
    assert(a2 == c2);
end
%% Initialize S_dw and S_db for gradient descent with RMS prop
[S_dW, S_db] = initialize_parameters_with_zero(layer_dims);
%% Defining activations for every layer
% Last layer has activation sigmoid
% All layers before it has activation relu
for i = 1:num_layers
    if i ~= num_layers
        if mod(i,2) > -1
            activation{i} = 'relu';
        else
            activation{i} = 'sigmoid';
        end
    else
        activation{i} = 'sigmoid';
    end
end

%% Start the training loop
for iter = 1 : 15000*2
    iter
    for k = 1 : length(x_train_mini_batch)
        k
        
        if drop_out == 1
                        disp('no L2');

            for layer = 1:length(layer_dims)
                prob_matrix{layer} = rand(layer_dims(layer), batch_size);
                if layer == 1
                    prob_matrix{layer} = prob_matrix{layer} < keep_prob_init;
                elseif layer == length(layer_dims)
                    prob_matrix{layer} = prob_matrix{layer} > -1;    % Everything equal to 1 for output layer
                else
                    prob_matrix{layer} = prob_matrix{layer} < keep_prob;
                end
            end
            %% Forward propagation of the network
            [A,Z] = forward_propagation_drop_out(x_train_mini_batch{k},W,b,activation,prob_matrix,keep_prob_init,keep_prob);
            %% Computing cost from last layer activation
            cost(k,iter) = compute_cost(y_train_mini_batch{k},A);
            if isnan(cost(iter))
                break;
            end
            %% Backward propagation
            [dW ,db] = backward_propagation_drop_out(x_train_mini_batch{k},y_train_mini_batch{k},A,Z,W,activation,prob_matrix,keep_prob_init,keep_prob);
        else
            disp('L2');
            [A,Z] = forward_propagation(x_train_mini_batch{k},W,b,activation);
            cost(k,iter) = compute_cost(y_train_mini_batch{k},A) + compute_cost_L2_regularisation_part(W,lambda);
            if isnan(cost(iter))
                break;
            end
            [dW, db] = backward_propagation(x_train_mini_batch{k},y_train_mini_batch{k},A,Z,W,activation);
            dW = back_prop_L2(dW,W,lambda);
        end
        %% Update Parameters - normal gradient descent
%             [W,b] = update_parameters(W, b, learning_rate, dW, db);
        %% Update Parameters with momentum
        [V_dW, V_db] =  momentum_avg(V_dW,dW,V_db,db,iter);
        [W,b] = update_parameters(W,b,learning_rate,V_dW,V_db);
         W = max_normalization(W,c);
         b = max_normalization(b,c);
    
%     %% Update Parameters with RMS Prop
%         [S_dW, S_db] = rmsProp_avg(S_dW,dW,S_db,db,iter);
%         [W,b] = update_parameters_rmsProp(W,b,learning_rate,dW,db,S_dW,S_db);
    %% Update Parameters with Adam
%         [V_dW, V_db] =  momentum_avg(V_dW,dW,V_db,db,iter);
%         [S_dW, S_db] = rmsProp_avg(S_dW,dW,S_db,db,iter);
%         [W,b] = update_parameters_rmsProp(W,b,learning_rate,V_dW,V_db,S_dW,S_db);
    end
    %% end the training loop
end


%% plot cost
cost = cost(:);
figure;
plot(cost)
%% plot A
figure;
imagesc(A{end})
%%

x_test = x_train;
y_test = y_train;
%% Predict y
y_predict = zeros(size(y_train,1),size(x_test,2));
[A_main,~] = forward_propagation(x_test,W,b,activation);
A_predict = A_main{end};
for i = 1:size(A_predict,2)
    [~,dummy] = max(A_predict(:,i));
    y_predict(dummy,i) = 1;
end
%%

figure;
hold on
subplot(2,1,1)
imagesc(y_test)

subplot(2,1,2)
imagesc(y_predict)

%% error

error = (sum(abs(y_test - y_predict)));
error = length(find(error > 0))/ size(y_test,2);


