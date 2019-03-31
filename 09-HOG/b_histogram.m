% train the SVM classifier and obtain histogram of distances from points to the hyperplane
% detector parameters don't affect this

close all
clear
run('vlfeat/toolbox/vl_setup')

data_path = 'data/'; %change if you want to work with a network copy
train_path_pos = fullfile(data_path, 'caltech_faces/Caltech_CropFaces'); %Positive training examples. 36x36 head crops
non_face_scn_path = fullfile(data_path, 'train_non_face_scenes'); %We can mine random or hard negatives from here

%The faces are 36x36 pixels, which works fine as a template size. You could
%add other fields to this struct if you want to modify HoG default
%parameters such as the number of orientations, but that does not help
%performance in our limited test.
feature_params = struct('template_size', 36, 'hog_cell_size', 6);
% SVM parameters
conf.svm.C = 1;
conf.svm.biasMultiplier = 1;
conf.svm.solver = 'sgd';

% train a SVM classifier
features_pos = get_positive_features( train_path_pos, feature_params );
num_negative_examples = 20000; %Higher will work strictly better, but you should start with 10000 for debugging
features_neg = get_random_negative_features( non_face_scn_path, feature_params, num_negative_examples);
[w,b] = classifier_training(features_pos,features_neg,conf);

% draw histograms for negative and positive confidences
positive_confidences = features_pos*w + b;
negative_confidences = features_neg*w + b;
ecdf(positive_confidences)
ecdf(negative_confidences)