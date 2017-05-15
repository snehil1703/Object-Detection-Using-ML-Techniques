function data = resize_image(data,output_type,new_r,new_c)

for j = 1:length(data)
    numImages = length(data(j).image);
    fprintf(1,'Processing food category %s with %d images.\n', data(j).name...
        ,numImages);
    for k = 1:numImages
        img = data(j).image{k};
        if class(img) == 'uint8'
            % if uint8, convert to double
            img = im2double(img);
        end
        
        [r,c,~] = size(img);
        r_sel = sort(randi(r,[new_r,1]));
        c_sel = sort(randi(c,[new_c,1]));
        switch output_type
            case 'grey'
                img = rgb2gray(img);
                data(j).resizeImg{k} = img(r_sel,c_sel);
            case 'color'
                data(j).resizeImg{k} = img(r_sel,c_sel,:);
        end
        %imshow(data(j).resizeImg{k});
        %drawnow;
    end
end