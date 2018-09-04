function [ output, mask ] = flip_noise( input, threshold )
%UNTITLED22 Summary of this function goes here
%   Detailed explanation goes here

%%flip
%mask = rand(size(input)) < threshold;
%output = xor(input, mask);

%%drop
mask = rand(size(input)) > threshold;
output = mask.*input;
end

