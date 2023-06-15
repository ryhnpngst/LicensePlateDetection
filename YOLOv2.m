trainingData = load('blacks.mat'); %perhatikan
trainingData.imageFilename = fullfile( ...
    trainingData.blacks.imageFilename); %perhatikan
rng(0);
trainingData2 = struct2table(trainingData);
shuffledIdx = randperm(height(trainingData2));
trainingData2 = trainingData2(shuffledIdx,:);
trainingData3 = trainingData2.blacks(:,:); %perhatikan
imds = imageDatastore (trainingData3.imageFilename); 
blds = boxLabelDatastore(trainingData3(:,2:end)); %2:end
ds = combine(imds, blds);
net = load('yolov2VehicleDetector.mat');
lgraph = net.lgraph;
lgraph.Layers
options = trainingOptions('adam',... %sgdm rmsprop adam
          'InitialLearnRate',0.0001,... %0.0001
          'Verbose',true,...
          'MiniBatchSize',8,... % 8 16
          'MaxEpochs',285,... %130 200 300
          'Shuffle','never',...
          'VerboseFrequency',30,...
          'CheckpointPath',tempdir);
[detectorYOLOv2,info] = trainYOLOv2ObjectDetector(ds,lgraph,options);

figure
plot(info.TrainingLoss)
grid on
xlabel('Number of Iterations')
ylabel('Training Loss')
plot(info.TrainingRMSE)