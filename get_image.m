function trainData = get_image(trainDir)
    trainData = dir(trainDir);
    trainData = rmfield(trainData,{'folder','date','bytes','isdir','datenum'});
    trainData = trainData(3:end) % remove . and ..
    %trainData(1).images = {}

    for j = 1:length(trainData)
        % Source: https://www.mathworks.com/matlabcentral/newsreader/view_thread/307759
        trainData(j).class = j;
        imgDir = strcat(trainDir,'/',trainData(j).name)
        filePattern = fullfile(imgDir, '*.jpg')
        jpegFiles = dir(filePattern);
        for k = 1:length(jpegFiles)
            baseFileName = jpegFiles(k).name;
            fullFileName = fullfile(imgDir, baseFileName);
            fprintf(1, 'Now reading %s\n', fullFileName);
            imageArray = imread(fullFileName);
            trainData(j).image{k} = imageArray;
            %imshow(trainData(1).image{k});  % Display image.
            %drawnow; % Force display to update immediately.
        end
    end
%     trainData
%     trainData.name
%     imshow(trainData(1).image{1})
end