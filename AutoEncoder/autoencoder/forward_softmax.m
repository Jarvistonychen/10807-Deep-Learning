function [ output ] = forward_softmax( input )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

input_num_row = size(input,1);
output = exp(input)./repmat(sum(exp(input)),input_num_row,1);

end

