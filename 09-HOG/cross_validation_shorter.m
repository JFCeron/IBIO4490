% performs detections at different levels of n_scales_multiplier and r_overlap and reports goodness measures
close all
clear
run('vlfeat/toolbox/vl_setup')

data_path = 'data/'; %change if you want to work with a network copy
train_path_pos = fullfile(data_path, 'caltech_faces/Caltech_CropFaces'); %Positive training examples. 36x36 head crops
non_face_scn_path = fullfile(data_path, 'train_non_face_scenes'); %We can mine random or hard negatives from here
test_scn_path = fullfile(data_path,'test_scenes/test_jpg'); %CMU+MIT test scenes
validation_scn_path = fullfile(data_path,'test_scenes/validation'); %CMU+MIT test scenes
new_test_scn_path = fullfile(data_path,'test_scenes/new_test'); %CMU+MIT test scenes
label_path = fullfile(data_path,'test_scenes/ground_truth_bboxes.txt'); %the ground truth face locations in the test set

% feature extraction parameters
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

% evaluate detection schema at different parametrizations
all_n_scales_multiplier = 3:3:9;
all_r_overlap = 0.3:0.3:0.3;
precision = zeros(size(all_n_scales_multiplier,2), size(all_r_overlap,2));
recall = zeros(size(all_n_scales_multiplier,2), size(all_r_overlap,2));
for n_scales_multiplier = all_n_scales_multiplier
	fprintf('___________Processing scale multiplier: %d___________\n\n',n_scales_multiplier);
	for r_overlap = all_r_overlap
		fprintf('___________Processing overlap ratio: %d___________\n\n',r_overlap);
		% run the detector with these parameters in the validation set
		[bboxes, confidences, image_ids] = run_detector(validation_scn_path, w, b, feature_params, n_scales_multiplier, r_overlap);
		% evaluate its performance
		[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = evaluate_detections(bboxes, confidences, image_ids, label_path);
		% those measurements include validation+test set
		validation_ids = ismember(gt_ids, image_ids);
		gt_ids = gt_ids(validation_ids);
		gt_isclaimed = gt_isclaimed(validation_ids);
		tp = tp(validation_ids);
		fp = fp(validation_ids);

		% store precision and recall
		i = find(all_n_scales_multiplier==n_scales_multiplier);
		j = find(all_r_overlap==r_overlap);
		precision(i,j) = sum(tp)/sum(tp+fp);
		recall(i,j) = sum(gt_isclaimed)/size(gt_isclaimed, 1);
	end
end
csvwrite('precision_shorter.csv', precision);
csvwrite('recall_shorter.csv', recall);