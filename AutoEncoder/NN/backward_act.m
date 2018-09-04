function [ gradient ] = backward_act( forward_in )
%UNTITLED13 Summary of this function goes here
%   Detailed explanation goes here
% in_size = size(forward_in);
% batch_size = in_size(2);
% row_num = in_size(1);
% gradient = zeros(row_num,batch_size);
% for iter=1:batch_size
%     gradient(:,iter) = diag(sigmoid_func(forward_in(:,iter)) .* (1-sigmoid_func(forward_in(:,iter))))*pre_gradient(:,iter);
% end
gradient = sigmoid_func(forward_in) .* (1-sigmoid_func(forward_in));
end

