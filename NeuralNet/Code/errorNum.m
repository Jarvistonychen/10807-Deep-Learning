function [ error_num ] = errorNum( output, label )
%UNTITLED19 Summary of this function goes here
%   Detailed explanation goes here

[~, ind] = max(output);
error_num = sum(ind ~= label);
end

