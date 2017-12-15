%% Script does the following: 
% 1. Outputs the ground truth segmentation, i.e. true states from input PCG. 
% 2. Features are extracted from true state and PCG.  
% 3. Extracted features used for PCG classification
% 4. Classification performance compared between true state and predicted states

close all;
clear all;

data_dir = [pwd filesep];
features_truth_file = 'features_gt';
features_lrhsmm_file = 'features_lrhsmm';
label_file = 'classes_lrhsmm';

if exist([data_dir features_truth_file '.csv'], 'file') == 2
    delete([data_dir features_truth_file '.csv']);
end
if exist([data_dir features_lrhsmm_file '.csv'], 'file') == 2
    delete([data_dir features_lrhsmm_file '.csv']);
end
if exist([data_dir label_file '.csv'], 'file') == 2
    delete([data_dir label_file '.csv']);
end

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
training_indices = data_indices(1:792);
train_recordings = example_data.example_audio_data(training_indices);
train_annotations = example_data.example_annotations(training_indices,:);

% pick out remaining examples for testing segmentation
test_index = data_indices(1:792);
test_recordings = example_data.example_audio_data(test_index);
test_annotations = example_data.example_annotations(test_index,:);

% true classification results for test data
test_answers = example_data.binary_diagnosis(test_index);


%% Train the HMM:

[B_matrix, pi_vector, total_obs_distribution] = trainSpringerSegmentationAlgorithm(train_recordings,train_annotations,springer_options.audio_Fs, false);

save B_matrix pi_vector total_obs_distribution

%% Run the HMM on test recording:

% number of test examples
numPCGs = length(test_recordings);

for PCGi = 1:numPCGs
    
    PCG_audio = test_recordings{PCGi};
    
    % HMM predicted state assignments
    [assigned_states] = runSpringerSegmentationAlgorithm(PCG_audio, springer_options.audio_Fs, B_matrix, pi_vector, total_obs_distribution, false);
    
    % ECG assigned state labels (reference)
    S1_locations = test_annotations{PCGi,1}; %ECG S1 marker
    S2_locations = test_annotations{PCGi,2}; %ECG S2 marker
    
    [PCG_Features, featuresFs] = getSpringerPCGFeatures(PCG_audio, springer_options.audio_Fs);
    PCG_states = labelPCGStates(PCG_Features(:,1),S1_locations, S2_locations, featuresFs);
    
    % Upsample true states to match array length of assigned states
    ratio = floor(length(assigned_states)./length(PCG_states));
    
    if (ratio > 1)
        rep_mat = repmat(PCG_states', ratio, 1); 
        true_states = rep_mat(:);
    else
        fprintf('Error: Assigned and PCG_state sample ratio is less than 1 %d \n', ratio);
        true_states = 0;
    end
    
    if (length(true_states) ~= length(assigned_states))
        fprintf('Error: Length Mismatch - true %d, assigned %d \n', length(true_states), length(assigned_states));
    end
        
    
    % Plotting assigned and true states:
    if(false)
        figure('Name','Assigned vs Labeled states to PCG');
        t1 = (1:length(PCG_audio))./springer_options.audio_Fs;
        t2 = (1:length(PCG_Features))./featuresFs;
        plot(t1, PCG_audio, 'k');
        hold on;
        plot(t2, PCG_states, 'b-');
        plot(t1, assigned_states, 'r--');
        plot(t1, true_states, 'g-');
        xlabel('Time (s)');
        legend('Audio data','Labeled States','Predicted States', 'True States');
    end
    
    % Extract features based on assigned and true states   
    assigned_features  = extractFeaturesFromHsIntervals(assigned_states,PCG_audio);
    true_features = extractFeaturesFromHsIntervals(true_states,PCG_audio);  
    
    if(sum(assigned_features) == 0) %throw out too short examples
        %do nothing
        fprintf('Thrown out %d \n', PCGi);
    else
        % Save features to CSV file
        dlmwrite([data_dir features_truth_file '.csv'], true_features, 'delimiter',',','-append');
        dlmwrite([data_dir features_lrhsmm_file '.csv'], assigned_features, 'delimiter',',', '-append');
        
        % Save class labels to CSV file
        dlmwrite([data_dir label_file '.csv'], test_answers{PCGi}.*2 -1, 'delimiter',',', '-append');
    end
end

