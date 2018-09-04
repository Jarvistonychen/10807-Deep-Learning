function [ gradient ] = backward_bias( out_num_neuron, batch_size )
%UNTITLED15 Summary of this function goes here
%   Detailed explanation goes here
% gradient = zeros(out_num_neuron,out_num_neuron*batch_size);
%     for i=1:batch_size
%         gradient(:,(i-1)*out_num_neuron+1:i*out_num_neuron) = diag(ones(1,out_num_neuron));
%     end
gradient = ones(1,out_num_neuron*batch_size);
end

