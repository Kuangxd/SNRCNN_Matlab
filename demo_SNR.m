close all;
clear all;
tic;

%% read ground truth image
img  = imread('Real_infrared_image\001.png');
% img  = imread('Set14\zebra.bmp');

%% load model
model = 'x1.mat';

%% work on gray only
if size(img,3)>1
    img = rgb2ycbcr(img);
    img = img(:, :, 1);
end
img = im2double(img);
size_o = size(img);

%% add stripe noise
img_n = img + repmat(random('norm', 0, 0.005, 1, size_o(2)), size_o(1), 1);

%% SNRCNN
img_c = SNRCNN(model, img_n);

%% remove border (9-5-5)
img1 = img((1 + 4) : (size_o(1) - 4), (1 + 8) : (size_o(2) - 8));
img2 = img_c((1 + 4) : (size_o(1) - 4), (1 + 8) : (size_o(2) - 8));
img1 = uint8(img1 * 255);
img2 = uint8(img2 * 255);

%% compute PSNR
psnr_srcnn = compute_psnr(img1,img2);

%% show results
figure, imshow(img_n); title('Noisy image');
figure, imshow(img_c); title('SNRCNN Reconstruction');
toc;
