function [ gradient ] = back_loss_func( NN_out, input )
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here

%label column-wise. NN_out column-wise.
%Both are multiple columns (batch)
%gradient = input./NN_out - (1-input)./(1-NN_out);
gradient = NN_out - input;
end

