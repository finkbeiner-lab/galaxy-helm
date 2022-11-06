function scale_space_matlab = generate_scale_space(im_in, sigma)

[rows, cols] = size(im_in);

% Allocate space to store all images in scale space
%scale_space = zeros(rows, cols, length(sigma));
scale_space_matlab = zeros(rows, cols, length(sigma));

% Create all the Laplacian of Gaussian images
for k=1:length(sigma)
    
    % Create a Gaussian with the given sigma
    %gaussian = create_gaussian(sigma(k));
    % Take the laplacian of the kernel
    %lap_gauss = del2(gaussian);
        
    % Convolve the kernel with the image
    %convolved = conv2(double(im_in), double(lap_gauss), 'same');
    
    % Store the image in our scale space array
    %scale_space(:,:,k) = (sigma(k)^2) * convolved;
    
    % Construct the laplacian of gaussian for a given kernel size and sigma
    n = ceil(sigma(k)*3)*2+1;
    lap_gauss = fspecial('log', n, sigma(k));
    
    % Convolve the kernel with the image
    convolved = conv2(double(im_in), double(lap_gauss), 'same');
    
    % Store the image in our scale space array
    scale_space_matlab(:,:,k) = (sigma(k)^2) * convolved;
    
end

