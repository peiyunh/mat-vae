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

% two copies for easier implementation
net1 = dagnn.DagNN.loadobj(net);
net2 = dagnn.DagNN.loadobj(net);

net1.conserveMemory = false; 
net1.mode = 'test';
net1.removeLayer('h3'); 
net1.removeLayer('tanh3'); 
net1.removeLayer('h4'); 
net1.removeLayer('sigmoid'); 
net1.removeLayer('KLD'); 
net1.removeLayer('NLL'); 
net1.removeLayer('LB'); 

net2.conserveMemory = false; 
net2.mode = 'test';
net2.removeLayer('h1');
net2.removeLayer('tanh1');
net2.removeLayer('h2');
net2.removeLayer('split');
net2.removeLayer('sample');

data = load_data('mnist', 'valid');

m = 13;
n = 13;
for i = 1:size(data,4)
    input = data(:,:,:,randi(size(data,4),1,m));
    net1.eval({'input', input});
    z = net1.vars(net1.getVarIndex('z')).value;

    latent_morph = [];
    image_morph = [];
    morph = [];
    %z0 = z(:,:,:,1);
    %zN = z(:,:,:,2:end);
    for j = 2:m
        imt = zeros(1,1,784,n,'single');
        for k = 1:n
            imt(:,:,:,k) = (input(:,:,:,j) - input(:,:,:,j-1)) * (k-1)/(n-1) + input(:,:,:,j-1);
        end
        imt = permute(reshape(imt,28,28,[]),[2,1,3]);
        imt = gather(squeeze(imt));
        imorph = tile_image(imt);
        image_morph = cat(1, image_morph, imorph);
        
        zn = z(:,:,:,j);
        zt = zeros(1,1,size(z,3),n,'single');
        for k = 1:n
            zt(:,:,:,k) = (z(:,:,:,j) - z(:,:,:,j-1)) * (k-1)/(n-1) + z(:,:,:,j-1);
        end
        net2.eval({'z', zt});

        prob = net2.vars(net2.getVarIndex('prob')).value;
        prob = permute(reshape(prob,28,28,[]),[2,1,3]);
        prob = gather(squeeze(prob));
        lmorph = tile_image(prob);
        latent_morph = cat(1, latent_morph, lmorph);
    end

    imwrite(1-latent_morph, 'report/res/latent_morph.png');
    imwrite(1-image_morph, 'report/res/image_morph.png');
    pause;
end