close all;

%optim = 'ADAM';
%hiddenSizes = [28*28, 500, 20];
%sfx = sprintfc('%d', hiddenSizes);
%sfx = [sfx{1} '-' sfx{2} '-' sfx{3}]; 
%epoch = 100;
%load(['models/mnist-' sfx '-' optim '/net-epoch-' num2str(epoch) '.mat']);

modelPath = ...; % specify a model under ../models/
load(modelPath);
net = dagnn.DagNN.loadobj(net);

net.conserveMemory = false; 
net.mode = 'test';

data = load_data('mnist', 'valid');
for i = 1:size(data,4)
    input = data(:,:,:,i);
    net.eval({'input', input});
    subplot(1,3,1);
    imagesc(reshape(input,28,28)');
    axis image;
    caxis([0,1]);
    title('Input');

    prob = net.vars(net.getVarIndex('prob')).value;
    subplot(1,3,2);
    imagesc(reshape(prob,28,28)');
    axis image;
    caxis([0,1]);
    title('Probabilitiy');


    subplot(1,3,3);
    %sample = rand(size(prob)) <= prob;
    binary = prob >= 0.5; 
    imagesc(reshape(binary,28,28)');
    axis image;
    caxis([0,1]);
    title('Binary');

    pause;
end

