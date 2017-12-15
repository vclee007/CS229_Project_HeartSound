%% importing, normalizing and splitting data
data_dir = [pwd filesep];
features_file = 'features_train';
label_file = 'classes_train';
features_dev_file = 'features_val';
label_dev_file = 'classes_val';
addpath(pwd)
data = dlmread([data_dir features_file '.csv'], ',');
ytrue = dlmread([data_dir label_file '.csv'], ',');
data_test = dlmread([data_dir features_dev_file '.csv'], ',');
ytrue_test = dlmread([data_dir label_dev_file '.csv'], ',');

n = size(data, 2);
m_all = size(data, 1);
m_train = m_all;
% % Ratio of training examples
% r = 0.8;
% p = randperm(m_all);
% m_train = round(m_all*r);
% m_dev = m_all - m_train;
% train_idx = p(1:m_train);
% dev_idx = p(m_train+1:end);
% 
% % adding intercept to data X
% train_data = data(train_idx, :); 
% train_label = ytrue(train_idx);
% test_data = data(dev_idx, :);
% test_label = ytrue(dev_idx);

train_data = data; 
train_label = ytrue;
test_data = data_test;
test_label = ytrue_test;


%% importing, normalizing and splitting data
rng(20171115, 'twister');
% if (exist('train_data', 'var') ~= 1)
%     train_data = load('images_train.csv');
%     train_label = load('labels_train.csv');
%     test_data = load('images_test.csv');
%     test_label = load('labels_test.csv');
% end

r = 0.8;
n = size(train_data, 2);
c = 2;
train_size = round(r*m_train);
p = randperm(size(train_data, 1));

% converting labels to one hot encoding
y_train = zeros(c, size(train_label, 1));
y_train((train_label + 1)./2 + 1 + c * (0:size(train_label, 1)-1)') = 1;
y_train = y_train';

y_test = zeros(c, size(test_label, 1));
y_test((test_label + 1)./2 + 1 + c * (0:size(test_label, 1)-1)') = 1;
y_test = y_test';

X_train_all = train_data(p, :);
y_train_all = y_train(p, :);
X_train = X_train_all(1:train_size, :);
X_dev = X_train_all(train_size+1:end, :);
y_train = y_train_all(1:train_size, :);
y_dev = y_train_all(train_size+1:end, :);
X_test = test_data;

%normalize the data
if (0)
    avg = mean(mean(X_train));
    s = std(reshape(X_train, [], 1));

    X_train = (X_train - avg) / s;
    X_dev = (X_dev - avg) / s;
    X_test = (X_test - avg) / s;
end
    
if (true)
    X_train = (X_train - mean(X_train))./std(X_train);
    X_dev = (X_dev - mean(X_dev))./std(X_dev);
    X_test = (X_test - mean(X_test))./std(X_test);
end


%% training the network (without regularization)
m = train_size;
h1 = 10; % 300 units in hidden layer
h2 = 10;

% initialize the parameters
W1 = randn(h1, n);
b1 = zeros(h1, 1);
W2 = randn(h2, h1);
b2 = zeros(h2, 1);
W3 = randn(c, h2);
b3 = zeros(c, 1);

num_epoch = 50;
batch_size = 50;
num_batch = m / batch_size;
learning_rate = 5;
lambda = 0.001;
train_loss = zeros(num_epoch, 1);
dev_loss = zeros(num_epoch, 1);
train_accuracy = zeros(num_epoch, 1);
dev_accuracy = zeros(num_epoch, 1);

%% Your code here
% num_epoch = 5;
% batch_size = 1;
% num_batch = m/batch_size;
% train_loss = zeros(num_epoch, 1);
% dev_loss = zeros(num_epoch, 1);
% train_accuracy = zeros(num_epoch, 1);
% dev_accuracy = zeros(num_epoch, 1);

for epoch = 1:num_epoch
    learning_rate = learning_rate/(epoch).^1.0;
    for batch = 0:num_batch-1
        X_batch = X_train(1+batch_size*batch:batch_size*(batch+1),:);
        y_batch = y_train(1+batch_size*batch:batch_size*(batch+1),:);
        [h1_output, h2_output, prob, loss] = forward_prop2(X_batch, y_batch, W1, b1, W2, b2, W3, b3, lambda);
        [dW1, db1, dW2, db2, dW3, db3] = backward_prop2(X_batch, y_batch, W1, b1, W2, b2, W3, b3, lambda);
        W1 = W1 - learning_rate.*dW1;
        b1 = b1 - learning_rate.*db1;
        W2 = W2 - learning_rate.*dW2;
        b2 = b2 - learning_rate.*db2;
        W3 = W3 - learning_rate.*dW3;
        b3 = b3 - learning_rate.*db3;
    end
    [h1_train, h2_train, train_prob, train_loss_] = forward_prop2(X_batch, y_batch, W1, b1, W2, b2, W3, b3, lambda);
    [h1_dev, h2_dev, dev_prob, dev_loss_] = forward_prop2(X_dev, y_dev, W1, b1, W2, b2, W3, b3, lambda);
    train_loss(epoch,1) = train_loss_;
    dev_loss(epoch,1) = dev_loss_;
    [y_train_hat, i_train] = max(train_prob,[],2);
    [y_train_gt, i_train_gt] = max(y_batch,[],2);
    [y_dev_hat, i_dev] = max(dev_prob,[],2);
    [y_dev_gt, i_dev_gt] = max(y_dev,[],2);
    train_accuracy(epoch,1) = sum(i_train == i_train_gt)/batch_size;
    dev_accuracy(epoch,1) = sum(i_dev == i_dev_gt)/size(y_dev,1);
end
save params_reg W1 b1 W2 b2 W3 b3
% Your code end here
%% plotting and displaying results
figure(1);
plot(1:num_epoch, train_loss, 'r', ...
    1:num_epoch, dev_loss, 'b');
legend('training set', 'dev set');
xlabel('epochs');
ylabel('loss');
figure(2);
plot(1:num_epoch, train_accuracy, 'r', ...
    1:num_epoch, dev_accuracy, 'b');
legend('training set', 'dev set');
xlabel('epochs');
ylabel('accuracy');

%% Testing the model 
% load params
load params_reg

[~, ~, test_pred, ~] = forward_prop2(X_test, y_test, W1, b1, W2, b2, W3, b3, lambda);
[~, y_c_test_pred] = max(test_pred, [], 2);
[~, y_c_test] = max(y_test, [], 2);
test_accuracy = sum(y_c_test_pred == y_c_test) / size(y_test, 1);
fprintf('test set accuracy: %f \n', test_accuracy);
