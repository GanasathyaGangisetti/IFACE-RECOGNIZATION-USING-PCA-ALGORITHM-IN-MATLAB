clc; clear; close all;
dataset_folder = 'C:\Users\umama\OneDrive\Desktop\project_folder\faces_dataset';
img_height = 100;
img_width = 100;
num_components = 5;   % Top eigenfaces to use
threshold = 3000;      % Adjust as needed based on dataset

imaqreset;
try
    vid = videoinput('winvideo', 1);
    set(vid, 'ReturnedColorSpace', 'rgb');
    preview(vid);
    disp('? Adjust your face. Capturing in 5 seconds...');
    pause(5);
    img = getsnapshot(vid);
    closepreview(vid); delete(vid); clear vid;
    
    img_gray = rgb2gray(img);
    img_resized = imresize(img_gray, [img_height, img_width]);
    % Create tested capture folder if not exists
tested_folder = 'C:\Users\umama\OneDrive\Desktop\project_folder\test_faces';
if ~exist(tested_folder, 'dir')
    mkdir(tested_folder);
end

% Generate timestamp filename
timestamp = datestr(now, 'yyyy_mm_dd_HH_MM_SS');
filename = fullfile(tested_folder, ['capture_' timestamp '.bmp']);

% Save captured image
imwrite(img_resized, filename);

% Optional: Also save a fixed name for immediate recognition
imwrite(img_resized, 'manual_test.bmp');

disp(['? Live image captured and saved as: ' filename]);

    figure; imshow(img_resized); title('? Captured Test Image');
    disp('? Live image captured and saved.');
catch ME
    disp(['? Webcam error: ' ME.message]);
    return;
end
img_files = dir(fullfile(dataset_folder, '*.bmp'));
num_images = length(img_files);
if num_images == 0
    disp('? No dataset images found.');
    return;
end

data_matrix = zeros(img_height * img_width, num_images);
for i = 1:num_images
    img = imread(fullfile(dataset_folder, img_files(i).name));
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img_resized = imresize(img, [img_height, img_width]);
    img_vector = double(img_resized(:));
    data_matrix(:, i) = img_vector;
end
disp(['? Loaded ' num2str(num_images) ' images from dataset.']);
mean_face = mean(data_matrix, 2);
A = data_matrix - repmat(mean_face, 1, num_images);  % ? FIXED
L = A' * A;
[V, D] = eig(L);
eigvals = diag(D);
[sorted_vals, idx] = sort(eigvals); % ascending order
idx = flipud(idx);                  % flip to get descending order

V = V(:, idx);
eigvec_L = V(:, 1:num_components);

eigenfaces = A * eigvec_L;
for i = 1:num_components
    eigenfaces(:, i) = eigenfaces(:, i) / norm(eigenfaces(:, i));
end

projected_images = eigenfaces' * A;
disp('? PCA training complete. Ready for recognition.');
test_img = imread('manual_test.bmp');
if size(test_img, 3) == 3
    test_img = rgb2gray(test_img);
end
test_img = imresize(test_img, [img_height, img_width]);
test_vector = double(test_img(:)) - mean_face;
projected_test = eigenfaces' * test_vector;
diffs = projected_images - repmat(projected_test, 1, num_images);
distances = sqrt(sum(diffs.^2, 1));
[min_distance, recognized_index] = min(distances);

disp('--------------------------------------');
disp(['? Minimum Distance: ' num2str(min_distance)]);

if min_distance < threshold
    recognized_name = img_files(recognized_index).name;
    recognized_img = reshape(data_matrix(:, recognized_index), [img_height, img_width]);
    
    figure; imshow(uint8(recognized_img));
    title(['? Recognized as: ' recognized_name]);
    
    disp(['? Face recognized as: ' recognized_name]);
    disp('? Recognition successful.');
else
    figure; imshow(uint8(test_img));
    title('? No Match Found');
    
    disp('? Recognition failed: No match found within threshold.');
end

disp(['? Image Size: ' num2str(img_height) ' x ' num2str(img_width)]);
disp('--------------------------------------');
