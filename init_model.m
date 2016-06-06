% init_model.m
% n0: input size
% n1: hidden layer size
% n2: encoding size
function net = init_model(n0, n1, n2)

net = dagnn.DagNN();

% set up input size
net.meta.normalization.imageSize = [1,1,n0];

% first hidden layer
net = add_layer(net, 'h1', n0, n1, 'input', 'h1');

% add tanh 
net.addLayer('tanh1', dagnn.Tanh(), 'h1', 'h1t');

% second hidden layer
net = add_layer(net, 'h2', n1, 2*n2, 'h1t', 'h2');

% split layer
mapping = {1:n2, (n2+1):2*n2}; 
net.addLayer('split', dagnn.Split('childIds',mapping), 'h2', {'mu', 'logvar'});

% sample
net.addLayer('sample', dagnn.Sampler(), {'mu','logvar'}, 'z'); 

% third hidden layer
net = add_layer(net, 'h3', n2, n1, 'z', 'h3');

% add tanh
net.addLayer('tanh3', dagnn.Tanh(), 'h3', 'h3t');

% 4th hidden layer
net = add_layer(net, 'h4', n1, n0, 'h3t', 'h4');

% add sigmoid
net.addLayer('sigmoid', dagnn.Sigmoid(), 'h4', 'prob');

% KLD (KL divergence)
net.addLayer('KLD', dagnn.KLD(), {'mu', 'logvar'}, 'KLD');

% NLL (negative log likelihood)
net.addLayer('NLL', dagnn.NLL(), {'prob', 'input'}, 'NLL');
%net.addLayer('NLL', dagnn.Loss('loss','binarylog'),...
%             {'prob', 'input'}, 'NLL');

net.addLayer('LB', dagnn.LB(), {'KLD', 'NLL', 'input'}, 'LB'); 

% for DEBUG
%for i = 1:numel(net.vars)
%    net.vars(i).precious = 1; 
%end

% meta parameters
net.meta.inputSize = [1,1,n0];
net.meta.normalization.border = [0,0];
net.meta.normalization.interpolation = 'none';
net.meta.normalization.averageImage = [];
net.meta.normalization.keepAspect = true;
net.meta.augmentation.rgbVariance = zeros(0,3);
net.meta.augmentation.transformation = 'none';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function net = add_layer(net, name, in, out, inputs, outputs)
net.addLayer(name, init_block(in,out), inputs, outputs, ...
             {[name 'f'], [name, 'b']});
net = init_weight(net, [name 'f'], in, out, 1);
net = init_weight(net, [name 'b'], 1, out, 2);

function block = init_block(in, out)
block = dagnn.Conv('size',[1,1,in,out],'stride',1,'pad',0);

function net = init_weight(net, name, in, out, lr)
idx = net.getParamIndex(name);
% xavier 
net.params(idx).value = (rand(1, 1, in, out, 'single')*2 - 1)*sqrt(3/in);
% xavierimproved 
%net.params(idx).value = randn(1, 1, in, out, 'single')*sqrt(2/out);
net.params(idx).learningRate = lr;
