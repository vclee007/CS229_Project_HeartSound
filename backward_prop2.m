function [dW1, db1, dW2, db2, dW3, db3] = backward_prop2(X, y_onehot, W1, b1, W2, b2, W3, b3, lambda)
%% backward propagation for our 1 layer network
%% input parameters
% X is our m x n dataset, where m = number of samples, n = number of
% features
% y is the length m label vector for each sample
% W1 is our h1 x n weight matrix, where h1 = number of hidden units in
% layer 1
% b1 is the length h1 column vector of bias terms associated with layer 1
% W2 is the h2 x h1 weight matrix, where h2 = number of hidden units in
% layer 2
% b2 is the length h2 column vector of bias terms associated with the output
% W3 is the c x h2 weight matriCopy_of_main2x, where c = number of classes
% b3 is the length c column vector of bias terms associated with the output
%% output parameters
% returns the gradient of W1, b1, W2, b2 as dW1, db1, dW2, db2
%% Your code here
% Your code end here
[h1_output, h2_output, prob, loss] = forward_prop2(X, y_onehot, W1, b1, W2, b2, W3, b3, lambda);
m = size(X, 1);
dW1 = 1/m.*(W2'*((W3'*(prob'-y_onehot')).*h2_output'.*(1 - h2_output')).*h1_output'.*(1 - h1_output'))*X + 2*lambda.*W1;
db1 = 1/m.*sum(W2'*((W3'*(prob'-y_onehot')).*h2_output'.*(1 - h2_output')).*h1_output'.*(1 - h1_output'), 2);
dW2 = 1/m.*(W3'*(prob'-y_onehot')).*h2_output'.*(1 - h2_output')*h1_output + 2*lambda.*W2;
db2 = 1/m.*sum((W3'*(prob'-y_onehot')).*h2_output'.*(1 - h2_output'), 2);
dW3 = 1/m.*(prob'-y_onehot')*h2_output + 2*lambda.*W3;
db3 = 1/m.*sum((prob'-y_onehot'), 2);
end