close all;
n2 = 20;
optims = {'ADAM', 'RMSPROP', 'ADAGRAD', 'SGD'};
train_kld = [];
valid_kld = [];
train_nll = [];
valid_nll = [];
train_lb = [];
valid_lb = [];

for i = 1:numel(optims)
    d = fullfile('models', ['mnist-784-500-' num2str(n2) '-' optims{i}]);
    load(fullfile(d, 'net-epoch-150.mat'));
    train_kld = cat(2, train_kld, [stats.train.KLD]');
    valid_kld = cat(2, valid_kld, [stats.val.KLD]');
    train_nll = cat(2, train_nll, [stats.train.NLL]');
    valid_nll = cat(2, valid_nll, [stats.val.NLL]');
    train_lb = cat(2, train_lb, [stats.train.LB]');
    valid_lb = cat(2, valid_lb, [stats.val.LB]');
end

names = {};
for i = 1:numel(optims)
    names{end+1} = sprintf('train-%s', optims{i});
end
for i = 1:numel(optims)
    names{end+1} = sprintf('val-%s', optims{i});
end
co = [0 0 1;
      0 0.5 0;
      1 0 0;
      0 0.75 0.75];

figure;
set(gca, 'FontSize', 20);
hold on;
plot(train_kld, 'linestyle', '--', 'linewidth', 3);
ax = gca; ax.ColorOrderIndex = 1;
plot(valid_kld, 'linewidth', 3);
hold off;
legend(names, 'location', 'southeast', 'FontSize', 20);
xlabel('Epochs');
ylabel('KL Divergence');
print('-dpdf', ['report/res/kld-n' num2str(n2) '.pdf']);

figure;
set(gca, 'FontSize', 18);
hold on;
plot(train_nll, 'linestyle', '--', 'linewidth', 3);
ax = gca; ax.ColorOrderIndex = 1;
plot(valid_nll, 'linewidth', 3);
hold off;
legend(names, 'location', 'northeast', 'FontSize', 20);
xlabel('Epochs');
ylabel('Negative Log-Likelihood');
print('-dpdf', ['report/res/nll-n' num2str(n2) '.pdf']);

figure;
set(gca, 'FontSize', 18);
hold on;
plot(train_lb, 'linestyle', '--', 'linewidth', 3);
ax = gca; ax.ColorOrderIndex = 1;
plot(valid_lb, 'linewidth', 3);
hold off;
legend(names, 'location', 'southeast', 'FontSize', 20);
xlabel('Epochs');
ylabel('Lower bound');
print('-dpdf', ['report/res/lb-n' num2str(n2) '.pdf']);
