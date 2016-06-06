%optim = 'ADAM';
%hiddenSizes = [28*28, 500, 2];
%sfx = sprintfc('%d', hiddenSizes);
%sfx = [sfx{1} '-' sfx{2} '-' sfx{3}]; 
%epoch = 150;
%load(['models/mnist-' sfx '-' optim '/net-epoch-' num2str(epoch) '.mat']);

modelPath = ...; % specify a model under ../models/
load(modelPath);
net = dagnn.DagNN.loadobj(net);

net.conserveMemory = false; 
net.mode = 'test';

net.removeLayer('h1');
net.removeLayer('tanh1');
net.removeLayer('h2');
net.removeLayer('split');
net.removeLayer('sample');

sample_size = 100;
mu = zeros(1,1,hiddenSizes(end),sample_size);
sig = ones(1,1,hiddenSizes(end),sample_size);
eps = randn(1,1,hiddenSizes(end),sample_size,'single');
z = mu + sig.*eps;

net.eval({'z', z});

prob = net.vars(net.getVarIndex('prob')).value;
prob = gather(squeeze(prob));

prob = reshape(prob, [28,28,size(prob,2)]);
prob = permute(prob, [2,1,3]);

clf; 
%imagesc(tile_image(prob));
%axis image;
%axis off;
%caxis([0, 1]);
imwrite(1-tile_image(prob), ['report/res/sample-n' num2str(hiddenSizes(end)) '.png']);
