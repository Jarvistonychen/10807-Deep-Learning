function [ gradient ] = backward_weight_slides( pre_after_act)
%UNTITLED15 Summary of this function goes here
%   Detailed explanation goes here

% for j=1:output_num_neuron
%     gradient((j-1)*input_num_neuron+1:j*input_num_neuron,((1:batch_size)-1)*output_num_neuron+j) = pre_after_act;
% end

gradient = pre_after_act';

end



