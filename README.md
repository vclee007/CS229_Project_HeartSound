# CS229_Project_HeartSound
Application of Machine Learning Techniques for Heart Sound Recording Classification By Vincent Lee and Anatoly Yakovlev

Dataset and LR-HSMM starter code were provided by Physionet: https://www.physionet.org/challenge/2016/

The following are files provided by Physionet:

butterworth_high_pass_filter
butterworth_low_pass_filter.m
default_Springer_HSMM_options.m
expand_qt.m
getDWT.m
get_duration_distributions.m
getHeartRateSchmidt.m
get_PSD_feature_Springer_HMM.m
getSpringerPCGFeatures.m
Hilbert_Envelope.m
Homomorphic_Envelope_with_Hilbert.m
labelPCGStates.m
runSpringerSegmentationAlgorithm.m
schmidt_spike_removal.m
trainBandPiMatricesSpringer.m
trainSpringerSegmentationAlgorithm.m
viterbiDecodePCG_Springer.m
viterbi_Springer.c
viterbi_Springer.mexa64

The following are files provided by Physionet, but modified by us:

extractFeaturesFromHsIntervals.m

The following are files created by us:

backward_prop2.m		  
backward_prop.m
cross_entropy.m
evaluate_metric.m
k_means_compress.m
weighted_LR.m
sigmoid_func.m
forward_prop2.m
lr_model.m
softmax_func.m
forward_prop.m
main.m
main2.m
main3.m
normalise_signal.m
svm_train.m
svm_test.m
pca.m
run_LR_HSMM.m
error_analysis_LRHSMM.m

The following are data files:

features_lrhsmm.csv
features_train.csv
B_matrix.mat
pi_vector.mat
total_obs_distribution.mat
example_data.mat
features_gt.csv
classes_lrhsmm.csv
classes_val.csv			  
features_val.csv
classes_train.csv		  
