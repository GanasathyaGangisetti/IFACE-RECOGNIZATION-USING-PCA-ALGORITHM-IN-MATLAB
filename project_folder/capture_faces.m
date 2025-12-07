clc; clear; close all;

vid = videoinput('winvideo', 1);
set(vid, 'ReturnedColorSpace', 'rgb');

preview(vid);
disp('Adjust your face position, capturing will start in 5 seconds...');
pause(5);
closepreview(vid);

output_folder = 'C:\Users\umama\OneDrive\Desktop\project_folder\faces_dataset';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

num_images = 5;
session_id = datestr(now, 'yyyy_mm_dd_HH.MM.SS');

for i = 1:num_images
    img = getsnapshot(vid);
    gray_img = rgb2gray(img);
    resized_img = imresize(gray_img, [100 100]);
    
    %filename = sprintf('person01_%02d.bmp', i); % Using PNG
    filename = sprintf('person01_%s_%02d.bmp', session_id, i);
    imwrite(resized_img, fullfile(output_folder, filename));

    imshow(resized_img);
    title(['Captured Image #' num2str(i)]);
    pause(1);
end

delete(vid);
clear vid;

disp('Image capture complete!');
