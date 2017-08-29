function Rohit_Mudaliar(input)
    input_image =((imread(input)));
    input_image_double = im2double(input_image);
    
    %Reading in the template image
    template = im2double(imread('front_tilt.jpg'));
    input_gray = rgb2gray(input_image);

    figure;
    imshow(input_image)
    title('input image')

    %Extracting different channels
    R = input_image(:,:,1);
    G = input_image(:,:,2);
    B = input_image(:,:,3);

    %zeroed image of same size
    BW = zeros(size(R));

    %Iterating through every pixel
    for r = 1:size(R,1)
        for c = 1:size(R,2)

            %Classifying the skin color model
            if (R(r,c) > 95 && G(r,c) > 40 && B(r,c) > 20) && ...
                    (abs(R(r,c) - G(r,c)) > 15 && R(r,c) > G(r,c) && R(r,c) > B(r,c))

                %Finding the maximum pixel of all the channels.
                if R(r,c) > G(r,c)

                    if R(r,c) > B(r,c)
                        maxP = R(r,c);
                    else
                        maxP = B(r,c);

                    end

                else

                    if G(r,c) > B(r,c)
                        maxP = G(r,c);
                    else
                        maxP = B(r,c);
                    end

                end



                %Finding the minimum pixel of all the channels    
                if R(r,c) < G(r,c)

                    if R(r,c) < B(r,c)
                        minP = R(r,c);
                    else
                        minP = B(r,c);

                    end

                else

                    if G(r,c) < B(r,c)
                        minP = G(r,c);
                    else
                        minP = B(r,c);
                    end

                end

                %Condition for difference of max and min pixel of all three channels 
                if maxP - minP > 15
                    BW(r,c) = 1;
                end

            end % End of outermost if condition

        end % End of inner for loop

    end % End of outer for loop

    %Converting to ycbcr color space
    ycbcr = rgb2ycbcr(input_image);

    %Extracting different components of ycbcr color space
    y = ycbcr(:,:,1);
    cb = ycbcr(:,:,2);
    cr = ycbcr(:,:,3);


    BW1 = zeros(size(y));

    %Iterating through all the pixels of the image
    for r = 1:size(y,1)
        for c = 1:size(y,2)

            %Classification of skin color pixels 
            if y(r,c) > 80 && 100 < cb(r,c) && cb(r,c) < 130 && 135 < cr(r,c) && ...
                    cr(r,c) < 175
                BW1(r,c) = 1;
            end
        end
    end

    %Converting to hsv color space
    hsv = rgb2hsv(input_image);

    %Extracting different color channel from the hsv color space
    h = hsv(:,:,1);
    s = hsv(:,:,2);

    BW2 = zeros(size(h));
    %Iterating through all the pixels of the image
    for r = 1:size(h,1)
        for c = 1:size(h,2)

            %Classifying the pixel as a skin color
            if h(r,c) > 0 && h(r,c) < 0.25 && s(r,c) > 0.15 && s(r,c) < 0.9
                BW2(r,c) = 1;
            end
        end
    end



    %%%%%%%%%%%%%%%%%%%%%%%%before morph%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    figure;


    subplot(2,2,2);
    imshow(BW2);
    title('hsv face before morph');

    %figure;
    subplot(2,2,3);
    imshow(BW1);
    title('ycbcr face before morph');

    %figure;
    subplot(2,2,4);
    imshow(BW);
    title('rgb face before morph');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %Morphology to increase the foregrond
    BW = imdilate(BW,strel('sphere',5));
    BW1 = imdilate(BW1,strel('sphere',5));
    BW2 = imdilate(BW2,strel('sphere',5));

    %Fill the holes in the foreground
    BW = imfill(BW);
    BW1 = imfill(BW1);
    BW2 = imfill(BW2);

    %Erode the image and make foreground smaller
    BW = imerode(BW,strel('sphere',10));
    BW1 = imerode(BW1,strel('sphere',10));
    BW2 = imerode(BW2,strel('sphere',10));



    BW3 = zeros(size(BW2));
    %To find a common area of interest extracted from different channels.
    for r = 1:size(BW3,1)
        for c = 1:size(BW3,2)
            if BW(r,c) == 1 && BW1(r,c) == 1 && BW2(r,c) == 1

                BW3(r,c) = 1;
            end
        end
    end

    %Morph the image
    BW3 = imerode(BW3,strel('disk',5));
    %Ignoring the areas of the foreground which are smaller than certain pixels
    BW3 = bwareaopen(BW3,1000);


    %Zeroed image of size input_image
    temp_image = zeros(size(input_image_double));

    [separations,n_labels]=bwlabel(BW3);
    %separations = medfilt2(separations, [5 5]);
    %Running the for loop to identify different dice.
    for ii=1:n_labels

        %To identify the various objects segmented by bwlabel.
        region=separations==ii;

        %Get cordinates of the region
        [x, y] = find(region == 1);
        xmin = round(min(x),0);
        xmax = round(max(x),0);
        ymin = round(min(y),0);
        ymax = round(max(y),0);

        %Isolate the identified region
        target = input_gray(xmin:xmax,ymin:ymax);
        if size(template) < size(target)
            c = normxcorr2(template,target);
        end

        %Find max correlation coefficient
        maxc = max(c(:));

        %Detecting the eyes in an image
        small_eye = vision.CascadeObjectDetector('EyePairSmall');
        bbox = step(small_eye, target);

        %Detecting eyes in an image
        big_eye = vision.CascadeObjectDetector('EyePairBig');
        bbox1 = step(big_eye, target);

        %Detecting nose in an image
        nose = vision.CascadeObjectDetector('Nose');
        nose.MergeThreshold = 10;
        bbox2 = step(nose, target);

        if maxc > 0.3 & (size(bbox) > 0 | size(bbox1) > 0 | size(bbox2) > 0)

            %Cordinates for drawing the rectngle 
            rect = ([ymin,xmin, ymax-ymin, xmax-xmin]);

            %Set properties of Shape Inserter to draw a rectangle
            shapeInserter = vision.ShapeInserter('LineWidth',5,'BorderColor','Custom',...
                                'CustomBorderColor', uint8([255 0 0]));

            %Draw the rectangle on the given cordinates.
            out = shapeInserter(temp_image,rect);
            input_image_double = input_image_double+out;

        end

        pause(2);
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%Final outcome%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure;
    subplot(2,2,1);
    imshow(BW3);
    title('final image after morphology');

    subplot(2,2,2);
    imshow(BW2);
    title('hsv face after morphology');

    subplot(2,2,3);
    imshow(BW1);
    title('ycbcr face after morphology');

    subplot(2,2,4);
    imshow(BW);
    title('rgb face after morphology');

    figure;
    imshow(input_image_double);
    title('final image for bounding box');


end