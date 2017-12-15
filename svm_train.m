%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% svm_train.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% rand('seed', 123);

%% Read in data
include_val = 3;
pca = 0;

data_dir = [pwd filesep];
features_file = 'features_train';
label_file = 'classes_train';
features_val_file = 'features_val';
label_val_file = 'classes_val';

if (pca)
    features_file = [features_file, '_pca'];
    features_val_file = [features_val_file, '_pca'];
end

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
    X_val = data_val;
    y_val = ytrue_val;
    p = randperm(m);
    m_train = round(m);
    m_dev = size(data_val,1);
    train_idx = p;
    X_dev = X_val;
    y_dev = y_val;
    X_train = data(train_idx, :);
    y_train = ytrue(train_idx);
elseif (include_val == 2) 
    % Mix all data together and then separate into train and val
    X_val = data_val;
    y_val = ytrue_val;
    X_train = data;
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
    X_val = data_val;
    y_val = ytrue_val;
    X_train = data;
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
    X_dev = data(dev_idx, :);
    y_dev = ytrue(dev_idx);
    X_train = data(train_idx, :);
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
    X_dev = data_gt;
    y_dev = ytrue_lrhsmm;
    m_dev = size(y_dev, 1);
end

m_train_size = m_train;
m_test = m_dev;
Xtrain = X_train;
Xtest = X_dev;
ytrain = y_train;
ytest = y_dev;
% randomize order of dataset
data_indices = randperm(m_train_size, m_train_size); 

%% Unit normalization per feature:
if (false)
    Xtrain = Xtrain./repmat(sqrt(sum(Xtrain.^2,1)),m_train,1);
    Xtest = Xtest./repmat(sqrt(sum(Xtest.^2,1)),m_test,1);
end

%% Generate matrices for SVM RBF Kernel
% trainSize = [50, 100, 200, m_train_size];
trainSize = m_train_size;
% trainSize = 50;
% lambda_list = 1/m_train.*[fliplr([2, 4, 8, 16, 64, 256]).^-1, [1, 4, 16, 64, 128]];
lambda_list = 1/(m_train*4);
% tau_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
% tau_list = [1/8, 1/4, 1/2, 1, 2, 4, 8];
tau_list = [3];
% tau_list = [64];

% Find Swept Hyperparameter and plot error rate versus the parameter
param = [sum(abs(diff(trainSize))), sum(abs(diff(tau_list))), ...
    sum(abs(diff(lambda_list)))];
param_idx = find(param ~= 0);
if (param_idx == 1)
    tau_list = tau_list.*ones(1,length(trainSize));
    lambda_list = lambda_list.*ones(1,length(trainSize));
    param_swept = trainSize;
elseif (param_idx == 2)
    trainSize = trainSize.*ones(1,length(tau_list));
    lambda_list = lambda_list.*ones(1,length(tau_list));
    param_swept = tau_list;
elseif (param_idx == 3)
    trainSize = trainSize.*ones(1,length(lambda_list));
    tau_list = tau_list.*ones(1,length(lambda_list));
    param_swept = lambda_list;
end

jj = 0; 
error = zeros(size(trainSize));
% rand('seed', 123);

for num_train = trainSize
    clear Xtrain squared_X_train tau avg_alpha;
    train_index = data_indices(1:num_train);
    m_train = num_train;
    Xtrain = data(train_index,:);
    ytrain = ytrue(train_index);
    jj = jj + 1; 

    squared_X_train = sum(Xtrain.^2, 2);
    gram_train = Xtrain * Xtrain';
    tau = tau_list(jj);

    % Get full training matrix for kernels using vectorized code.
    Ktrain = full(exp(-(repmat(squared_X_train, 1, m_train) ...
                        + repmat(squared_X_train', m_train, 1) ...
                        - 2 * gram_train) / (2 * tau^2)));

    lambda = lambda_list(jj);
    num_outer_loops = 30;
    alpha = zeros(m_train, 1);

    avg_alpha = zeros(m_train, 1);
    Imat = eye(m_train);

    count = 0;
    for ii = 1:(num_outer_loops * m_train)
      count = count + 1;
      ind = ceil(rand * m_train);
      margin = ytrain(ind) * Ktrain(ind, :) * alpha;
      g = -(margin < 1) * ytrain(ind) * Ktrain(:, ind) + ...
          m_train * lambda * (Ktrain(:, ind) * alpha(ind));
      % g(ind) = g(ind) + m_train * lambda * Ktrain(ind,:) * alpha;
      alpha = alpha - g / sqrt(count);
      avg_alpha = avg_alpha + alpha;
    end
    avg_alpha = avg_alpha / (num_outer_loops * m_train);

    error(jj) = svm_test(Xtrain, squared_X_train, tau, avg_alpha, Xtest, ytest);
end

%% Find Swept Hyperparameter and plot error rate versus the parameter

h1 = plot(param_swept, error.*100);
% Formatting plot
fontsize = 14;
label(1) = xlabel('Training Size','Fontsize', 14);
label(2) = ylabel('Test Error (%)');
label(3) = title('Test Error vs Training Set Size for SVM');
set(gca,'FontSize', 14);
set(label(1), 'fontsize', fontsize);
set(label(2), 'Fontsize', fontsize);
set(label(3), 'Fontsize', fontsize);
set(h1,'linewidth',2,'color','b');

