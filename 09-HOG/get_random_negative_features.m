% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);

% how many examples could we draw from each image with our method?
samplesXimage = zeros(num_images,1);
rescalesXimage = zeros(num_images,1);
for i = 1:num_images
	img_info = imfinfo(strcat(non_face_scn_path,'/',image_files(i).name));
	width = img_info.Width;
	height = img_info.Height;
	min_dim = min(width,height);
	% how many times can I rescale this image at a factor of 2 before going below
	% the template size in at least one dimension?
	rescalesXimage(i) = floor(log2(min_dim/feature_params.template_size));
	for j = 0:rescalesXimage(i)
		samplesXimage(i) = samplesXimage(i)+4^j;
	end
end
max_samples = sum(samplesXimage);

% if we want less than the maximum
if (num_samples < max_samples)
	% decrease the number of samples per image. At least 1
	samplesXimage = floor((num_samples/max_samples)*samplesXimage);
	samplesXimage = max(samplesXimage,1);
end

% now make that number of negative samples
features_neg = zeros(sum(samplesXimage), (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
added = 0;
for i = 1:num_images
	img = imread(strcat(non_face_scn_path,'/',image_files(i).name));
	gray = rgb2gray(img);
	% draw the requested number of samples from the image
	added_here = 0;
	
	% first a template-sized version of it
	first = imresize(gray, [feature_params.template_size,feature_params.template_size]);
	hog = vl_hog(single(first), feature_params.hog_cell_size);
	features_neg(added+1,:) = reshape(hog,1,(feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
	added = added + 1;
	added_here = added_here + 1;
	
	% now some scaled versions. Biggest is level 0. Skip the smallest
	for level = (rescalesXimage(i)-1):-1:0
		% do we need to add more negatives from this image?
		if added_here >= samplesXimage(i)
			break
		end
		rescaled = imresize(gray,1/(2^level));
		% pick out 4^(pyr_height - level) negatives
		num_samples_level = 4^(rescalesXimage(i)-level);
		for negative = 1:num_samples_level
			% randomly select the window location in the image
			x = randi(size(rescaled, 1) - feature_params.template_size);
			y = randi(size(rescaled, 2) - feature_params.template_size);
			% do it
			non_face = rescaled(x:(x+feature_params.template_size-1),y:(y+feature_params.template_size-1));
			hog = vl_hog(single(non_face), feature_params.hog_cell_size);
			features_neg(added+1,:) = reshape(hog,1,(feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
			added = added + 1;
			added_here = added_here + 1;
			if added_here >= samplesXimage(i)
				break
			end
		end
	end
end