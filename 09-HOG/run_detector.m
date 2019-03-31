% Starter code prepared by James Hays for CS 143, Brown University
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    run_detector(test_scn_path, w, b, feature_params, n_scales_multiplier, r_overlap)
% n_scales_multiplier : the number of scales considered at each image depends on template to image ratio *n_scales_multiplier
% r_overlap : sliding window overlap ratio

% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

% number of overlapping pixels per dimension in sliding window schema
overlap = floor(feature_params.template_size*r_overlap);
% threshold for considering an SVM face detection positive
face_threshold = 0;

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));
%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

% for every test scene
for i = 1:length(test_scenes)
    % this scene's face detections, confidences + scene id
    bboxes_scene = zeros(0,4);
    confidences_scene = zeros(0,1);
    image_ids_scene = cell(0,1);
    %prints the current image in the loop and if it is not in grey scale then turns it to grey scale
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    % img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end

    % we will pass a sliding window detector at scales ranging from full image to 36xX
	min_dim = min(size(img));
	template2img_ratio = feature_params.template_size/min_dim;
    % the number of scales considered at each image depends on template to image ratio
    n_scales = ceil(n_scales_multiplier/template2img_ratio);
	scales = 2:(template2img_ratio-2)/n_scales:template2img_ratio;
	for scale = scales
		smaller = imresize(img, scale);

        % obtain features at sliding windows
        % define sliding window coordinates
        n_x_ticks = floor((size(smaller,2) - overlap)/(feature_params.template_size - overlap));
        x_ticks = 1 + (0:(n_x_ticks-1))*(feature_params.template_size - overlap);
        n_y_ticks = floor((size(smaller,1) - overlap)/(feature_params.template_size - overlap));
        y_ticks = 1 + (0:(n_y_ticks-1))*(feature_params.template_size - overlap);
        % obtain features at each window
        windows_features = zeros(n_y_ticks*n_x_ticks, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
        % store top left corners to later write bounding boxes
        top_left_corners = zeros(n_y_ticks*n_x_ticks, 2);
        added = 0;
        for x = x_ticks
            for y = y_ticks
                sliding_window = smaller(y:(y+feature_params.template_size-1), x:(x+feature_params.template_size-1));
                window_descriptor = vl_hog(single(sliding_window),feature_params.hog_cell_size);
                windows_features(added + 1,:) = reshape(window_descriptor, 1, []);
                top_left_corners(added + 1,:) = [x y];
                added = added + 1;
            end
        end

        % classify windows
        w = reshape(w, [], 1);
        b = reshape(b, [], 1);
        confidences_scale = windows_features*w + b;
        positives = confidences_scale > face_threshold;
        % obtain bounding boxes and confidences for the image at this scale
        confidences_scale = confidences_scale(positives);
        bboxes_scale = [top_left_corners(positives,:) top_left_corners(positives,:)+feature_params.template_size];
        % normalize bounding boxes by scale before adding them to the scene bboxes list
        bboxes_scene      = [bboxes_scene;      floor(bboxes_scale/scale)];
        confidences_scene = [confidences_scene; confidences_scale];
	end
    % filter the scene's bboxes by non-maximum supression
    [is_maximum] = non_max_supr_bbox(bboxes_scene, confidences_scene, size(img));
    % add this scene's bboxes to the list of all bboxes in the test set
    bboxes      = [bboxes;      bboxes_scene(is_maximum,:)];
    confidences = [confidences; confidences_scene(is_maximum,:)];
    image_ids   = [image_ids;   transpose(repelem({test_scenes(i).name}, sum(is_maximum)))];
end