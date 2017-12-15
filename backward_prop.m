function [dW1, db1, dW2, db2] = backward_prop(X, y_onehot, W1, b1, W2, b2, lambda)
%% backward propagation for our 1 layer network
%% input parameters
% X is our m x n dataset, where m = number of samples, n = number of
% features
% y is the length m label vector for each sample
% W1 is our h1 x n weight matrix, where h1 = number of hidden units in
% layer 1
% b1 is the length h1 column vector of bias terms associated with layer 1
% W2 is the c x h1 weight matrix, where c = number of classes
% b2 is the length h2 column vector of bias terms associated with the output
%% output parameters
% returns the gradient of W1, b1, W2, b2 as dW1, db1, dW2, db2
%% Your code here
% Your code end here
[h_output, prob, loss] = forward_prop(X, y_onehot, W1, b1, W2, b2, lambda);
% dW1 = (W2'*(prob'-y_onehot')).*h_output'.*(ones(size(h_output'))-h_output')*X;
% db1 = (W2'*(prob'-y_onehot')).*h_output'.*(ones(size(h_output'))-h_output');
% dW2 = (prob'-y_onehot')*h_output;
% db2 = prob'-y_onehot';
m = size(X, 1);
dW1 = 1/m.*(W2'*(prob'-y_onehot')).*h_output'.*(ones(size(h_output'))-h_output')*X + 2*lambda.*W1;
db1 = 1/m.*sum((W2'*(prob'-y_onehot')).*h_output'.*(ones(size(h_output'))-h_output'), 2);
dW2 = 1/m.*(prob'-y_onehot')*h_output + 2*lambda.*W2;
db2 = 1/m.*sum((prob'-y_onehot'), 2);

end