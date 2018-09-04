function [ loss ] = loss_func( input,label )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

%input and label are column-major
loss = sum(-log(input).*label);
loss = loss'; %column vector

end

