function [ output ] = forward( input, weights, bias )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

batch_size = size(input,2);
output = weights*input+repmat(bias,1,batch_size);

end

