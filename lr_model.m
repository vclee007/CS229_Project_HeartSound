% compute gradient given X, y, and theta
compute_grad = @(X, y, theta) (-1 / size(X, 1)) * X' * (y ./(1 + exp((X * theta).* y)));

%% training for dataset A
disp('============= LR model ============');
include_val = 3;

data_dir = [pwd filesep];
features_file = 'features_train';
label_file = 'classes_train';
features_val_file = 'features_val';
label_val_file = 'classes_val';
addpath(pwd)
data = dlmread([data_dir features_file '.csv'], ',');
ytrue = dlmread([data_dir label_file '.csv'], ',');
data_val = dlmread([data_dir features_val_file '.csv'], ',');
ytrue_val = dlmread([data_dir label_val_file '.csv'], ',');
n = size(data, 2);
m = size(data, 1);

% Ratio of training examples
if (include_val == 1)
    % Use train and validation datasets
    X_val = [ones(size(data_val,1), 1), data_val];
    y_val = ytrue_val;
    p = randperm(m);
    m_train = round(m);
    m_dev = size(data_val,1);
    train_idx = p;
    X_dev = X_val;
    y_dev = y_val;
    X_train = [ones(m_train, 1), data(train_idx, :)];
    y_train = ytrue(train_idx);
elseif (include_val == 2) 
    % Mix all data together and then separate into train and val
    X_val = [ones(size(data_val,1), 1), data_val];
    y_val = ytrue_val;
    X_train = [ones(m, 1), data];
    y_train = ytrue;
    X_train_all = [X_train; X_val];
    y_train_all = [y_train; y_val];
    m_train_all = size(X_train_all, 1);
    
    r = 0.8;
    p = randperm(m_train_all);
    m_train = round(m_train_all*r);
    m_dev = m_train_all - m_train;
    train_idx = p(1:m_train);
    dev_idx = p(m_train+1:end);
    X_dev = X_train_all(dev_idx, :);
    y_dev = y_train_all(dev_idx);
    X_train = X_train_all(train_idx, :);
    y_train = y_train_all(train_idx);
elseif (include_val == 3) 
    % Mix all data together and then separate into train and val 
    % Both train and val have 50% mix of classes
    X_val = [ones(size(data_val,1), 1), data_val];
    y_val = ytrue_val;
    X_train = [ones(m, 1), data];
    y_train = ytrue;
    X_train_all = [X_train; X_val];
    y_train_all = [y_train; y_val];
    m_train_all = size(X_train_all, 1);
    r = 0.8;
    
    idx_p = find(y_train_all == 1);
    idx_n = find(y_train_all == -1);
    m_balanced = min(length(idx_p), length(idx_n));
    p_p = randperm(m_balanced);
    p_n = randperm(length(idx_n));
    m_train_p = round(m_balanced * r);
    m_dev_p = m_balanced - m_train_p;
    train_idx_p = idx_p(p_p(1:m_train_p));
    train_idx_n = idx_n(p_n(1:m_train_p));
    train_idx = [train_idx_p; train_idx_n];
    train_idx = train_idx(randperm(length(train_idx)));
    X_train = X_train_all(train_idx, :);
    y_train = y_train_all(train_idx, 1);
    m_train = size(X_train, 1);
    
    dev_idx_p = idx_p(p_p(m_train_p+1:m_train_p + m_dev_p));
    dev_idx_n = idx_n(p_n(m_train_p+1:m_train_p + m_dev_p));
    dev_idx = [dev_idx_p; dev_idx_n];
    dev_idx = dev_idx(randperm(length(dev_idx)));
    X_dev = X_train_all(dev_idx, :);
    y_dev = y_train_all(dev_idx, 1);
    m_dev = size(X_dev, 1);
else
    % Use only train dataset for both train and validation
    r = 0.7;
    p = randperm(m);
    m_train = round(m*r);
    m_dev = m - m_train;
    train_idx = p(1:m_train);
    dev_idx = p(m_train+1:end);
    X_dev = [ones(m_dev, 1), data(dev_idx, :)];
    y_dev = ytrue(dev_idx);
    X_train = [ones(m_train, 1), data(train_idx, :)];
    y_train = ytrue(train_idx);
end

% Error Analysis Flag
error_analysis = 0;
lrhsmm_mode = 0;
if(lrhsmm_mode == 1) 
    features_file_gt = 'features_lrhsmm'; 
else
    features_file_gt = 'features_gt';
end
label_file_lrhsmm = 'classes_lrhsmm'; %label for dev
if (error_analysis)
    data_gt = dlmread([data_dir features_file_gt '.csv'], ',');
    ytrue_lrhsmm = dlmread([data_dir label_file_lrhsmm '.csv'], ',');
    X_dev = [ones(size(data_gt, 1), 1), data_gt];
    y_dev = ytrue_lrhsmm;
    m_dev = size(y_dev, 1);
end

%% Unit normalization per feature:
if (false)
    X_train = X_train./repmat(sqrt(sum(X_train.^2,1)),m_train,1);
    X_dev = X_dev./repmat(sqrt(sum(X_dev.^2,1)),m_dev,1);
end

feature_subset = [1, 5, 18:21];
feature_subset = [1:1:n+1];
X_train = X_train(:,feature_subset);

% [idx_pos] = find(y>0);
% [idx_neg] = find(y<0);
% figure(3); scatter(X(idx_pos,2),X(idx_pos,3),'r+')
% figure(3); hold on; scatter(X(idx_neg,2),X(idx_neg,3),'bo'); hold off;
% title('Dataset A', 'fontsize', fontsize);
%%
% training a logistic model
theta = zeros(size(X_train, 2), 1);
learning_rate = 0.1;
close all;
figure(1); 
for i = 1:10^9
    prev_theta = theta;
    grad = compute_grad(X_train, y_train, theta);
    theta = theta - learning_rate.*(i.^-2) * grad;
    if mod(i, 10000) == 0
        fprintf('finished iteration %d \n', i); 
        y_train_hat = 1./(1+exp(-X_train*theta));
        y_train_hat = y_train_hat >= 0.5;
        y_train_hat = 2.*y_train_hat - 1;
        train_err = sum(abs(y_train - y_train_hat))/(2*length(y_train));
        fprintf('Training error %e \n', train_err);
%         fprintf('Gradient %e \n', grad);
%         fprintf('X*theta %e, %e \n', max(X*theta.*y), min(X*theta.*y));
%         fprintf('theta %e \n', max(theta));
         
         figure(1); h1 = scatter(i, train_err); hold on;
         figure(2); h2 = scatter(i.*ones(size(theta)), theta); hold on;
%         h1 = scatter(i, grad(1),'r'); 
%         h1 = scatter(i, grad(2),'b*');
%         h1 = scatter(i, grad(3),'k'); 
    end
    if norm(theta - prev_theta) < 10^-10
        fprintf('converged in %d iterations \n', i-1);
        break;
    end
end
hold off;

%% Solution Using Normal Equations
% Training error is better but test error seems to be worse
if (false)
    theta2 = (X_train'*X_train)^-1*X_train'*y_train;
    theta1 = theta;
    theta = theta2;
end
%% Prediction
threshold = 0.5;
y_train_hat = 1./(1+exp(-X_train*theta));
y_train_hat = y_train_hat > threshold;
y_train_hat = 2.*y_train_hat - 1;
train_err = sum(abs(y_train - y_train_hat))/(2*length(y_train));

% Performance Metric for Training
disp('============= Training Performance ============');
evaluate_metric(y_train_hat, y_train);

% Cross Validation Prediction
y_dev_hat = 1./(1+exp(-X_dev(:,feature_subset)*theta));
y_dev_hat = y_dev_hat > threshold;
y_dev_hat = 2.*y_dev_hat - 1;
dev_err = sum(abs(y_dev - y_dev_hat))/(2*length(y_dev));

% Performance Metric for Dev
disp('============= Dev Performance ============');
evaluate_metric(y_dev_hat, y_dev);

% fprintf('Training Err: %2.1f %%\n', train_err*100);
% fprintf('Test Err: %2.1f %%\n', dev_err*100);




% hold off;
% set(gca, 'YScale', 'log')
% % Formatting plot
% fontsize = 14;
% label(1) = xlabel('Iteration number','Fontsize', 14);
% label(2) = ylabel('Gradient');
% label(3) = title('Gradient vs Iteration Number for Dataset A');
% set(gca,'FontSize', 14);
% set(label(1), 'fontsize', fontsize);
% set(label(2), 'Fontsize', fontsize);
% set(label(3), 'Fontsize', fontsize);
% set(h1,'linewidth',2);%,'color','b');
% 
% yhat = (1./(1+exp(-X*theta))>0.5)*2-1;
% fprintf('Number of errors : %d\n', sum(y-yhat~=0));
% x1 = min(X(:,2)):1e-1:max(X(:,2));
% x2 = -theta(1)/theta(3) - theta(2)/theta(3).*x1;
% figure(3); hold on; plot(x1, x2, 'r'); hold off;

%% 
function [train_idx, dev_idx] = separateSets(m, r)
% Separate data into training and validation data sets
% m - total number of examples
% r - ratio of training examples
    m_train = round(m*r);
    m_dev = m - m_train;
    dev_idx = [1:1:m];
    train_idx = randperm(m, m_train);
    dev_idx(train_idx) = [];
end
