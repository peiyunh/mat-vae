%optim = 'ADAM';
%hiddenSizes = [28*28, 500, 2];
%sfx = sprintfc('%d', hiddenSizes);
%sfx = [sfx{1} '-' sfx{2} '-' sfx{3}]; 
%epoch = 100;
%load(['models/mnist-' sfx '-' optim '/net-epoch-' num2str(epoch) '.mat']);

modelPath = ...;
load(modelPath); % specify a model under ../models/
net = dagnn.DagNN.loadobj(net);

net.conserveMemory = false; 
net.mode = 'test';

net.removeLayer('h1');
net.removeLayer('tanh1');
net.removeLayer('h2');
net.removeLayer('split');
net.removeLayer('sample');

ny = 20;
nx = 20;
%Ys = icdf('normal', linspace(0,1,ny+2), 0, 1); Ys = Ys(2:end-1);
%Xs = icdf('normal', linspace(0,1,nx+2), 0, 1); Xs = Xs(2:end-1);
Ys = linspace(-3,3,ny); 
Xs = linspace(-3,3,nx); 
[yy,xx] = meshgrid(Ys, Xs);
z = cat(2,yy(:),xx(:))';
z = single(reshape(z, [1, 1, size(z)]));

net.eval({'z', z});

prob = net.vars(net.getVarIndex('prob')).value;
prob = gather(squeeze(prob));

prob = reshape(prob, [28,28,size(prob,2)]);
prob = permute(prob, [2,1,3]);

%imagesc(tile_image(prob));
%axis image;
%caxis([0, 1]);
imwrite(1-tile_image(prob), 'report/res/manifold.png');
