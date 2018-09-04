function [ gradient ] = backward_weight_faster( pre_after_act, input_num_neuron, output_num_neuron )
%UNTITLED15 Summary of this function goes here
%   Detailed explanation goes here

% for j=1:output_num_neuron
%     gradient((j-1)*input_num_neuron+1:j*input_num_neuron,((1:batch_size)-1)*output_num_neuron+j) = pre_after_act;
% end
enlarge_pre_after_act = repmat(pre_after_act,output_num_neuron,1);
gradient = reshape(enlarge_pre_after_act, input_num_neuron,[]);

end


