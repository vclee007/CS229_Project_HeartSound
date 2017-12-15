%% Script does the following: 
% 1. Train segmentation algorithm, which is used to generate assigned states from input PCG. 
% 2. Features are extracted from state and PCG.  
% 3. Features used for PCG classification

close all;
clear all;
debug = 0;

%% Load the default options:
% These options control options such as the original sampling frequency of
% the data, the sampling frequency for the derived features and whether the
% mex code should be used for the Viterbi decoding:
springer_options = default_Springer_HSMM_options;

if (springer_options.use_mex)
    if (~(exist('viterbi_Springer.mexa64', 'file') == 3))
        mex viterbi_Springer.c
    end
end

%% Load the audio data and the annotations:
% These are 792 example PCG recordings, downsampled to 1000 Hz, with
% annotations of the R-peak and end-T-wave positions.
load('example_data.mat');

%% Split the data into train and test sets:

% randomize order of dataset
data_indices = randperm(792, 792); 

% pick out 633 of 792 examples for training segmentation
training_indices = data_indices(1:720);
train_recordings = example_data.example_audio_data(training_indices);
train_annotations = example_data.example_annotations(training_indices,:);

% pick out remaining examples for testing segmentation
test_index = data_indices(721:792);
test_recordings = example_data.example_audio_data(test_index);
test_annotations = example_data.example_annotations(test_index,:);


%% Train the HMM:

[B_matrix, pi_vector, total_obs_distribution] = trainSpringerSegmentationAlgorithm(train_recordings,train_annotations,springer_options.audio_Fs, false);

save('B_matrix.mat', 'B_matrix');
save('pi_vector.mat', 'pi_vector');
save('total_obs_distribution.mat', 'total_obs_distribution');
%% Run the HMM on test recording:

% metric values for use later, each element is counter for one heart state,
% S1, systole, S2, diastole
TP = [0, 0, 0, 0]; %true positive
FP = [0, 0, 0, 0]; %false positive
FN = [0, 0, 0, 0]; %false negative

% number of test examples
numPCGs = length(test_recordings);

if (debug)
    numPCGs = 10;
end

for PCGi = 1:numPCGs
    
    PCG_audio = test_recordings{PCGi};
    
    % HMM predicted state assignments
    [assigned_states] = runSpringerSegmentationAlgorithm(PCG_audio, springer_options.audio_Fs, B_matrix, pi_vector, total_obs_distribution, false);
    
    % ECG assigned state labels (reference)
    S1_locations = test_annotations{PCGi,1}; %ECG S1 marker
    S2_locations = test_annotations{PCGi,2}; %ECG S2 marker
    
    [PCG_Features, featuresFs] = getSpringerPCGFeatures(PCG_audio, springer_options.audio_Fs);
    
    PCG_states = labelPCGStates(PCG_Features(:,1),S1_locations, S2_locations, featuresFs);
    
    % Downsample assigned states to match array length of PCG state labels
    pred_len = length(assigned_states);
    label_len = length(PCG_states);
    if (pred_len > label_len)
        sample_step = floor(pred_len / label_len);
        downsampled_assigned_states = assigned_states(1:sample_step:end);
    else
        downsampled_assigned_states = assigned_states;
    end
    
    % Plotting assigned and predicted states:
    if(false)
        figure('Name','Assigned vs Labeled states to PCG');
        t1 = (1:length(PCG_audio))./springer_options.audio_Fs;
        t2 = (1:length(PCG_Features))./featuresFs;
        plot(t1, PCG_audio, 'k');
        hold on;
        plot(t2, PCG_states, 'b-');
        plot(t1, assigned_states, 'r--');
        plot(t2, downsampled_assigned_states, 'g-');
        xlabel('Time (s)');
        legend('Audio data','Labeled States','Predicted States', 'Downsampled Predicted States');
    end
    
    % 100 ms delta window
    delta = 0.1*featuresFs; %in terms of samples
    
    % find all index locations when heart state transition occurs
    trans_index_ref = find(diff(PCG_states) ~= 0);
    trans_index_pred = find(diff(downsampled_assigned_states) ~= 0);
    
    %
    if (PCG_states(1) == downsampled_assigned_states(1)) % only do comparison if they start out at same state
        ptr = mod(PCG_states(1), 4) + 1;
        for index = 1:length(trans_index_ref)
            ref_phase_index = trans_index_ref(index);
            
            left_range = ref_phase_index - delta;
            right_range = ref_phase_index + delta;
            
            N1 = length(find((trans_index_pred < right_range) & (trans_index_pred > left_range)));
            
            if (N1 > 0)
                TP(ptr) = TP(ptr) + 1;
                FP(ptr) = FP(ptr) + N1 - 1;
            elseif (N1 == 0)
                FN(ptr) = FN(ptr) + 1;
            end
            ptr = mod(ptr, 4) + 1;
        end
    else
        fprintf('Error: Heart phase mismatch in segmentation for data no. %d \n', test_index(PCGi));
    end
    
end

%% Calculate HMM evaluation metrics

Se = (TP./(TP + FN));
P_plus = (TP./(TP + FP));
Acc = (TP./(TP + FP + FN));
F1 = (2.*Se.*P_plus)./(Se + P_plus);

% Metrics in Percentage
Percentage_Se = 100*Se
Percentage_P_plus = 100*P_plus
Percentage_Acc = 100*Acc
Percentage_F1 = 100*F1

%% Load RECORDS from training sets
dataset = {'training-a', 'training-b', 'training-c', 'training-d', ... 
    'training-e', 'training-f', 'validation'};

dest_dir = [pwd filesep];
addpath(pwd)

class_file = 'classes_train';
features_file = 'features_train';
val_class_fn = 'classes_val';
val_feat_fn = 'features_val';
% fn = dir([data_dir class_file '.csv']);
if exist([dest_dir features_file '.csv'], 'file') == 2
  delete([dest_dir features_file '.csv']);
end
if exist([dest_dir class_file '.csv'], 'file') == 2
  delete([dest_dir class_file '.csv']);
end
if exist([dest_dir val_feat_fn '.csv'], 'file') == 2
  delete([dest_dir val_feat_fn '.csv']);
end
if exist([dest_dir val_class_fn '.csv'], 'file') == 2
  delete([dest_dir val_class_fn '.csv']);
end

for ii = 1 : length(dataset)
    data_dir = [pwd filesep '/..' filesep dataset{ii} filesep];
    addpath(pwd)
    display(['Processing ', dataset{ii}]);

    if dataset{ii} == 'validation'
        features_file = val_feat_fn;
        class_file = val_class_fn;
    end

    fid = fopen([data_dir 'RECORDS'],'r');
    if(fid ~= -1)
        RECLIST = textscan(fid,'%s');
    else
        error(['Could not open ' data_dir 'RECORDS for scoring. Exiting...'])
    end
    fclose(fid);
    RECORDS = RECLIST{1};
    formatSpec = '%s%f';
    T = readtable([data_dir 'REFERENCE.csv'],'Delimiter',',', ...
        'Format',formatSpec);
    T_cell = table2cell(T);
    rec_fnames = T_cell{1:end,1};

    %% Run the HMM on all recordings from database

    numPCGs = size(T_cell,1);

    if (debug)
        numPCGs = 10;
    end

    for PCGi = 1:numPCGs
        filename = T_cell{PCGi};
        [PCG, Fs1] = audioread([data_dir filename '.wav']); % Load data
        PCG_resampled = resample(PCG,springer_options.audio_Fs,Fs1); % resample to springer_options.audio_Fs (1000 Hz)    
        [assigned_states] = runSpringerSegmentationAlgorithm(PCG_resampled, springer_options.audio_Fs, B_matrix, pi_vector, total_obs_distribution, false);
        features  = extractFeaturesFromHsIntervals(assigned_states,PCG_resampled);

        % Save features to CSV file
        dlmwrite([dest_dir features_file '.csv'], features, 'delimiter',',','-append');

        % Save classifcation output to CSV file
    %     dlmwrite([data_dir class_file '.csv'], classifyResult, 'delimiter',',','-append');
    end

    %% Check error rate and record labels into classes.csv
    % yhat = csvread([data_dir 'classes.csv']);
    ytrue = double(T.Var2);
    label_file = class_file;
    dlmwrite([dest_dir label_file '.csv'], ytrue, 'delimiter',',','-append');
    % error_count = sum(ytrue-yhat ~= 0);
end % going through datasets