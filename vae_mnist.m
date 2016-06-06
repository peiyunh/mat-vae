function [net, info] = vae_mnist(varargin)
%% configure
opts.optim = 'SGD';
opts.gpus = [1];
opts.hiddenSizes = [28*28, 500, 2];
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = sprintfc('%d', opts.hiddenSizes);
sfx = [sfx{1} '-' sfx{2} '-' sfx{3}];
opts.expDir = fullfile('models', ['mnist-' sfx '-' opts.optim]);
[opts, varargin] = vl_argparse(opts, varargin) ;

%% optimization parameters
opts.train = struct() ;
opts.train.gpus = opts.gpus; 
opts.train.numEpochs = 150;
opts.train.batchSize = 250;
opts.train.derOutputs = {'NLL', 1, 'KLD', 1};

%% load data 
train = load_data('mnist', 'train');
valid = load_data('mnist', 'valid');
data = cat(4, train, valid);

imdb.images.data = data;
imdb.images.set = vertcat(ones(size(train, 4), 1), ...
                          2*ones(size(valid, 4), 1));

%% initialize model 
rng(0);
net = init_model(opts.hiddenSizes(1), opts.hiddenSizes(2), opts.hiddenSizes(3));

%% start training
switch opts.optim
  case 'SGD'
    trainfn = @sgd_train;
    opts.train.learningRate = 0.0005;
  case 'ADAGRAD'
    trainfn = @adagrad_train;
    opts.train.learningRate = 0.01;
  case 'RMSPROP'
    trainfn = @rmsprop_train;
    opts.train.learningRate = 0.001;
  case 'ADAM'
    trainfn = @adam_train;
    opts.train.learningRate = 0.001;
end

[net, info] = trainfn(net, imdb, getBatch(opts), 'expDir', opts.expDir, ...
                      opts.train) ;

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.train.gpus)) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images} ;