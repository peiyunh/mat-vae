optims = {'SGD', 'ADAGRAD', 'RMSPROP', 'ADAM'} ;
hiddenSizes = {[28*28,500,2], [28*28,500,20], [28*28,500,200]};
for i = 1:numel(hiddenSizes)
    for j = 1:numel(optims)
        vae_mnist('optim', optims{j}, 'hiddenSizes', hiddenSizes{i});
    end
end
