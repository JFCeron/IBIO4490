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

% calibrate b using ground truth annotations in the validation set
%validation image files
val_img_files = dir( fullfile( validation_scn_path, '*.jpg' ));
n_val = length(val_img_files);
% ground truth annotations
fid = fopen(label_path);
gt_info = textscan(fid, '%s %d %d %d %d');
fclose(fid);
gt_ids = gt_info{1,1};
gt_bboxes = [gt_info{1,2}, gt_info{1,3}, gt_info{1,4}, gt_info{1,5}];
gt_bboxes = double(gt_bboxes);
% store the features of the validation faces
val_faces_features = zeros(1, size(features_pos,2));
for i = 1:n_val
	val_img = imread(strcat(validation_scn_path,'/',val_img_files(i).name));
	img_bboxes = gt_bboxes(ismember(gt_ids,val_img_files(i).name),:);
	% some bounding boxes run outside the image
	img_bboxes = max(1, img_bboxes);
	img_bboxes(:, [2 4]) = min(size(val_img, 1), img_bboxes(:, [2 4]));
	img_bboxes(:, [1 3]) = min(size(val_img, 2), img_bboxes(:, [1 3]));
	% store this images' validation bboxes
	for j = 1:size(img_bboxes,1)
		cutout = val_img(img_bboxes(j,2):img_bboxes(j,4), img_bboxes(j,1):img_bboxes(j,3));
		cutout = imresize(cutout, [feature_params.template_size feature_params.template_size]);
		hog = vl_hog(single(cutout), feature_params.hog_cell_size);
		val_faces_features = [val_faces_features; reshape(hog,1,(feature_params.template_size / feature_params.hog_cell_size)^2 * 31)];
	end
end

% how far are true faces from our hyperplane?
ecdf(val_faces_features*w+b)