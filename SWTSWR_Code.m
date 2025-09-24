clc; clear; close all;

% ============ STEP 1: Load and Preprocess Input Images ============
[file, pathname] = uigetfile('*','Load Image 1'); cd(pathname);
y = imread(file);
[file, pathname] = uigetfile('*','Load Image 2'); cd(pathname);
z = imread(file);

image1 = double(y); if size(image1,3)==3, image1 = rgb2gray(image1); end
image2 = double(z); if size(image2,3)==3, image2 = rgb2gray(image2); end

if ~isequal(size(image1), size(image2)), error('Images must be same size'); end

% ============ STEP 2: SWT Decomposition (Level 3) ============
level = 3;
[cA1, cH1, cV1, cD1] = swt2(image1, level, 'db1');
[cA2, cH2, cV2, cD2] = swt2(image2, level, 'db1');

% Fuse Approximation Coefficients
SF1 = sqrt(mean2(diff(cA1(:,:,level),1,1).^2) + mean2(diff(cA1(:,:,level),1,2).^2));
SF2 = sqrt(mean2(diff(cA2(:,:,level),1,1).^2) + mean2(diff(cA2(:,:,level),1,2).^2));
total_SF = SF1 + SF2;
if total_SF > 0
    w1 = SF1 / total_SF; w2 = SF2 / total_SF;
else
    w1 = 0.5; w2 = 0.5;
end
fused_cA = cA1; 
fused_cA(:,:,level) = w1 * cA1(:,:,level) + w2 * cA2(:,:,level);

% Fuse High-Frequency Subbands
window_size = 5; kernel = ones(window_size);
fused_cH = zeros(size(cH1)); fused_cV = zeros(size(cV1)); fused_cD = zeros(size(cD1));
for l = 1:level
    eH1 = sqrt(imfilter(cH1(:,:,l).^2, kernel, 'symmetric'));
    eH2 = sqrt(imfilter(cH2(:,:,l).^2, kernel, 'symmetric'));
    maskH = eH1 >= eH2; fused_cH(:,:,l) = maskH .* cH1(:,:,l) + (~maskH) .* cH2(:,:,l);

    eV1 = sqrt(imfilter(cV1(:,:,l).^2, kernel, 'symmetric'));
    eV2 = sqrt(imfilter(cV2(:,:,l).^2, kernel, 'symmetric'));
    maskV = eV1 >= eV2; fused_cV(:,:,l) = maskV .* cV1(:,:,l) + (~maskV) .* cV2(:,:,l);

    eD1 = sqrt(imfilter(cD1(:,:,l).^2, kernel, 'symmetric'));
    eD2 = sqrt(imfilter(cD2(:,:,l).^2, kernel, 'symmetric'));
    maskD = eD1 >= eD2; fused_cD(:,:,l) = maskD .* cD1(:,:,l) + (~maskD) .* cD2(:,:,l);
end

% Reconstruct fused SWT image
fused_image_swt = iswt2(fused_cA, fused_cH, fused_cV, fused_cD, 'db1');

% ============ STEP 3: WLS Post-Processing ============
lambda_wls = 1.2;
fused_image_wls = imguidedfilter(fused_image_swt, 'NeighborhoodSize', [9 9], ...
                                  'DegreeOfSmoothing', lambda_wls);

% ============ STEP 4: Saliency-Based Final Fusion ============
sal_swt = imgradient(fused_image_swt, 'sobel');
sal_wls = imgradient(fused_image_wls, 'sobel');
total_sal = sal_swt + sal_wls + eps;
w_swt = sal_swt ./ total_sal;
w_wls = sal_wls ./ total_sal;
fused_final = w_swt .* fused_image_swt + w_wls .* fused_image_wls;

% ============ STEP 5: Display Only Final Result ============
figure; imshow(mat2gray(fused_final));
title('Final Fused Image (Saliency)');

% ============ STEP 6: Compute Only Final Metrics ============
img1 = mat2gray(image1);
img2 = mat2gray(image2);
f_final = mat2gray(fused_final);

% Local helpers
calc_entropy = @(x) -sum(histcounts(x,256,'Normalization','probability') .* ...
                         log2(histcounts(x,256,'Normalization','probability') + eps));
calc_scd = @(x, y, f) abs(corr2(x, f) - corr2(y, f));
fmi = @(x, y) mutualinfo(double(edge(x, 'sobel')), double(edge(y, 'sobel')));

% Metrics
FMI_final = fmi(img1, f_final) + fmi(img2, f_final);
M_final   = mean(f_final(:));
FF_final  = mutualinfo(f_final,img1) + mutualinfo(f_final,img2);
SD_final  = std(f_final(:));
E_final   = calc_entropy(f_final);
SCD_final = calc_scd(img1, img2, f_final);

% Display Final Metrics
fprintf('\n========= FINAL FUSION METRICS =========\n');
fprintf('FMI           : %.4f\n', FMI_final);
fprintf('Mean (M)      : %.4f\n', M_final);
fprintf('Fusion Factor : %.4f\n', FF_final);
fprintf('Std. Dev (SD) : %.4f\n', SD_final);
fprintf('Entropy (E)   : %.4f\n', E_final);
fprintf('SCD           : %.4f\n', SCD_final);

% Save final fused output
imwrite(uint8(255 * mat2gray(fused_final)), 'fused_image_final_saliency.jpg');

% ============ MUTUAL INFORMATION FUNCTION ============
function MI = mutualinfo(img1, img2)
    img1 = uint8(255 * mat2gray(img1));
    img2 = uint8(255 * mat2gray(img2));
    jointHist = accumarray([double(img1(:))+1, double(img2(:))+1], 1, [256, 256]);
    jointProb = jointHist / sum(jointHist(:)) + eps;
    px = sum(jointProb, 2); py = sum(jointProb, 1);
    px_py = px * py + eps;
    MI = sum(jointProb(:) .* log2(jointProb(:) ./ px_py(:)));
end
