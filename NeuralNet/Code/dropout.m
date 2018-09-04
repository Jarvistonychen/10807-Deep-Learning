function [ output, mask ] = dropout( input, threshold )
%UNTITLED22 Summary of this function goes here
%   Detailed explanation goes here

mask = rand(size(input)) > threshold;
output = input .* mask;
end

