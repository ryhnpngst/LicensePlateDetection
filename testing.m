img = imread('moto1.jpg');
load 'detectorBlacks.mat'
[bboxes, scores] = detect(detectorYOLOv2,img);
if(~isempty(bboxes))
    img = insertObjectAnnotation(img,'rectangle',bboxes,scores,'Color','green');
end
figure
imshow(img)