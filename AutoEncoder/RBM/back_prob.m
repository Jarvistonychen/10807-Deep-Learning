function [ b_prob ] = back_prob( h, weights, c )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
batch_size = size(h,2);

b_prob = 1./(1+exp(-weights'*h-repmat(c,[1,batch_size])));


end

