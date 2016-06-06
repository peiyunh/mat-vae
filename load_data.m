function data = load_data(dataset, split)
if nargin < 2 || isempty(dataset) || isempty(split)
    dataset = 'mnist';
    split = 'train';
end

path = ['data/' dataset '.hdf5']; 
switch dataset
  case 'mnist'
    data = h5read(path, ['/x_' split]);
    data = single(data>=0.5); % binarization
    data = reshape(data, [1,1,size(data)]);
  case 'freyfaces'
    error('continuous input is not implemented');
end