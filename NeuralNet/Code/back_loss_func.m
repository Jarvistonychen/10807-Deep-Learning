function [ gradient ] = back_loss_func( NN_out, label )
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here

%label column-wise. NN_out column-wise.
%Both are multiple columns (batch)
gradient = NN_out - label;

end

