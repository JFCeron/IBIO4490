% Starter code prepared by James Hays for CS 143, Brown University
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    run_detector(test_scn_path, w, b, feature_params)
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

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);


%Starts the loop for every test scene(every test image)
for i = 1:length(test_scenes)
    
    %prints the current image in the loop and if it is not in grey scale then turns it to grey scale
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end

    %number of scales in the pyramid, every umage wll have the sliding window detector pass over a gaussian pyramid in order to detect faces of different scale
	n_scales = 100;
	% get bounding boxes for the test scenes

	  
	min_dim = min(size(img));
	% scales range from full image to 36xX
	template2img_ratio = feature_params.template_size/min_dim;
	scales = 1:(template2img_ratio-1)/n_scales:template2img_ratio;


	for scale = scales
		%image is rezized
	    smaller = imresize(img, scale);
	    %WindoSize of the sliding window, it is an hyperparameter
	    WindowSize =  feature_params.template_size ;
		% a function to retrieve window information
		aplicarHog = @(block_struct) vl_hog(single(img), feature_params.hog_cell_size)  ;
		% Matrix that evaluates the overlapped grid, i every cell excute a Hog detector
		WindowsArrayTemp2 = blockproc(smaller, [WindowSize WindowSize], aplicarHog, 'BorderSize', [floor(WindowSize/2) floor(WindowSize/2)], 'TrimBorder', false, 'PadPartialBlocks', false);
		[dim1 dim2] = size(WindowsArrayTemp2) ;
		%construct method for temporal lists that store the boxes and cofidences of positives for every pass trought a given scale
		cur_x_min_escala = [];
		cur_y_min_escala = [];
		cur_confidences_escala = [];
		%Loop over the WindowsArraTemp2 grid for doing the actual classification
		for i=1:dim1
			for j=1:dim2
				%Discrimination made by the SVM
				discriminacion = dot(reshape(WindowsArrayTemp2(i,j,:),[1,31]), w) - b;
				WindowsArrayTemp2(i,j) = discriminacion ;
				%Store the confidence anda top left corner for the positive detections
				if discriminacion > 0
					cur_x_min_escala = append(cur_x_min_escala,i*floor(WindowSize/2));
					cur_y_min_escala = append(cur_x_min_escala,i*floor(WindowSize/2));
					cur_confidences_escala = append(cur_confidences_escala, discriminacion);
				end 
			end
		end	

		%Array with the boxes coordinates
		cur_bboxes_escala = [cur_x_min_escala, cur_y_min_escala, cur_x_min_escala + WindowSize, cur_y_min_escala + WindowSize];

    	cur_image_ids(1:length(cur_confidences_escala),1) = {test_scenes(i).name};
    
    	%non_max_supr_bbox can actually get somewhat slow with thousands of
    	%initial detections. You could pre-filter the detections by confidence,
    	%e.g. a detection with confidence -1.1 will probably never be
   		%meaningful. You probably _don't_ want to threshold at 0.0, though. You
    	%can get higher recall with a lower threshold. You don't need to modify
   		 %anything in non_max_supr_bbox, but you can.
    	[is_maximum] = non_max_supr_bbox(cur_bboxes_escala, cur_confidences_escala, size(smaller));

    	cur_confidences_escala = cur_confidences_escala(is_maximum,:);
    	cur_bboxes_escala      = cur_bboxes_escala(     is_maximum,:);
    	cur_image_ids_escala   = cur_image_ids_escala(  is_maximum,:);
 				
    	%Adjusting boxes to real size
    	cur_bboxes = cur_bboxes_escala / scale; 
    	%Adding to return lists
    	bboxes      = [bboxes;      cur_bboxes];
    	confidences = [confidences; cur_confidences];
    	image_ids   = [image_ids;   cur_image_ids];





	    	size(smaller)
	end
end




