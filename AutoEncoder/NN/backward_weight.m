function [ gradient ] = backward_weight( pre_after_act, input_num_neuron, output_num_neuron )
%UNTITLED15 Summary of this function goes here
%   Detailed explanation goes here

%expand 2D weight matrix to 1D column vector. Row-wise.
batch_size = size(pre_after_act,2);
gradient=zeros(input_num_neuron*output_num_neuron, output_num_neuron*batch_size);
% for batch=1:batch_size
%     %gradient(:,(iter-1)*output_num_neuron+1:iter*output_num_neuron) = full(spdiags(repmat(pre_after_act(:,iter)',1,output_num_neuron)', 1-pre_after_act(:,iter)' , input_num_neuron+output_num_neuron-1, output_num_neuron) );
%     for j=1:output_num_neuron
%         gradient((j-1)*input_num_neuron+1:j*input_num_neuron,(batch-1)*output_num_neuron+j) = pre_after_act(:,batch);
%     end
% end

for j=1:output_num_neuron
    gradient((j-1)*input_num_neuron+1:j*input_num_neuron,((1:batch_size)-1)*output_num_neuron+j) = pre_after_act;
end

end

