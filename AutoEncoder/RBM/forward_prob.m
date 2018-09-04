function [ f_prob ] = forward_prob( x, weights, b )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
batch_size = size(x,2);
f_prob = 1./(1+exp(-weights*x-repmat(b,[1,batch_size])));
end

