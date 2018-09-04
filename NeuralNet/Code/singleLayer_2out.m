clc;
clear; 
temp_train_set = csvread('../digitstrain.txt');
valid_set = csvread('../digitsvalid.txt');
test_set = csvread('../digitstest.txt');

simDigit1 = 0;
simDigit2 = 6;

input_num_neuron = 784;
hidden1_num_neuron = 100;
output_num_neuron = 2;

train_set = temp_train_set(randperm(size(temp_train_set,1)),:);
train_label_single_ori = train_set(:,785);
train_label_single_array = vertcat(find(train_label_single_ori == 0), find(train_label_single_ori==simDigit2));
train_set_06 = train_set(train_label_single_array,:);
train_data = train_set_06(:,1:784);
train_label_single = train_set_06(:,785);
train_label_single_01 = train_label_single;
train_label_single_01(train_label_single_01 == simDigit2) = 1;
train_label = zeros(numel(train_label_single_array),output_num_neuron);
for iter=1:numel(train_label_single_array)
    train_label(iter,train_label_single_01(iter)+1) = 1;
end

%train_data = train_data/max(max(train_data));
valid_label_single_ori = valid_set(:,785);
valid_label_single_array = vertcat(find(valid_label_single_ori==0), find(valid_label_single_ori==simDigit2));
valid_set_06 = valid_set(valid_label_single_array,:);
valid_data = valid_set_06(:,1:784);
valid_label_single = valid_set_06(:,785);
valid_label_single_01 = valid_label_single;
valid_label_single_01(valid_label_single_01==simDigit2) = 1;
valid_label = zeros(numel(valid_label_single_array),output_num_neuron);
for iter=1:numel(valid_label_single_array)
    valid_label(iter,valid_label_single_01(iter)+1) = 1;
end

test_label_single_ori = test_set(:,785);
test_label_single_array = vertcat(find(test_label_single_ori==0), find(test_label_single_ori==simDigit2));
test_set_06 = test_set(test_label_single_array,:);
test_data = test_set_06(:,1:784);
test_label = test_set_06(:,785);
% test_label_single = test_set_06(:,785);
% test_label_single_01 = test_label_single;
% test_label_single_01(test_label_single_01==simDigit2) = 1;
% test_label = zeros(numel(test_label_single_array),2);


lr=0.01;
momentum=0.9;
w_decay = 0.0005; %0.0005;
dropout_prob = 0.5; %0.5;
epoch_num = 600;
batch_size = 1;
batch_num = numel(train_label_single_array)/batch_size;


%for runs=5:10
% for lr=[0.01,0.1]
%     for momentum=[0.5, 0.9]
%         for hidden1_num_neuron=[100,200]
%             for epoch_num=[200, 600]
%                 for w_decay=[0.0005, 0.001]
%                     for dropout_prob=[0.5, 0.9]
                        disp(['MNIST_2out_',num2str(lr),'_',num2str(momentum),'_',num2str(hidden1_num_neuron),'_',num2str(epoch_num),'_',num2str(w_decay),'_',num2str(dropout_prob)]);
                        %initialize weights
                        rand_bound1 = sqrt(6)/sqrt(hidden1_num_neuron+input_num_neuron);
                        rand_bound2 = sqrt(6)/sqrt(hidden1_num_neuron+output_num_neuron);

                        rand_weights1 = -rand_bound1 + 2*rand_bound1*rand(hidden1_num_neuron, input_num_neuron);
                        rand_weights2 = -rand_bound2 + 2*rand_bound2*rand(output_num_neuron, hidden1_num_neuron);
                        rand_bias1 = -rand_bound1 + 2*rand_bound1*rand(hidden1_num_neuron,1);%zeros(hidden1_num_neuron,1);
                        rand_bias2 = -rand_bound2 + 2*rand_bound2*rand(output_num_neuron,1);%zeros(output_num_neuron,1);

                        weights1 = rand_weights1;
                        weights2 = rand_weights2;
                        bias1 = rand_bias1;
                        bias2 = rand_bias2;

                        train_loss = zeros(1,epoch_num);
                        valid_loss = zeros(1,epoch_num);
                        train_error = zeros(1,epoch_num);
                        valid_error = zeros(1,epoch_num);
                        old_gradient_weights2 = zeros(output_num_neuron,hidden1_num_neuron);
                        old_gradient_weights1 = zeros(hidden1_num_neuron, input_num_neuron);
                        for epoch=1:epoch_num  
%                             rand_shuffle = randperm(size(temp_train_set,1));
%                             train_data = train_data(rand_shuffle,:);
%                             train_label = train_label(rand_shuffle,:);
                            %disp(['Epoch: ', num2str(epoch)]);
                            for batch=1:batch_num
                                input = train_data((batch-1)*batch_size+1:batch*batch_size,:)';
                                label = train_label((batch-1)*batch_size+1:batch*batch_size,:)';
                                %forward

                                out1 = forward(input, weights1, bias1);               
                                out1_act = forward_act(out1);
                                [out1_act, mask] = dropout(out1_act, dropout_prob);
                                out2 = forward(out1_act,weights2, bias2);
                                final_out = forward_softmax(out2);
                                final_error_num = errorNum(final_out,(train_label_single((batch-1)*batch_size+1:batch*batch_size)+1)');
                                final_loss = loss_func(final_out, label);     
                                train_loss(epoch) = train_loss(epoch) + sum(final_loss);
                                train_error(epoch) = train_error(epoch) + final_error_num;
                                %disp([num2str(batch), ': ', num2str(mean(final_loss))]);

                                %backward
                                back1 = back_loss_func(final_out, label);
                                %temp_gradient_weights2 = backward_weight_faster(out1_act,hidden1_num_neuron,output_num_neuron);
                                temp_gradient_weights2 = backward_weight_slides(out1_act);
                                %gradient_weights2 = backward_weight(out1_act,hidden1_num_neuron,output_num_neuron)*reshape(back1,[],1);
                                %gradient_weights2 = sum(reshape(temp_gradient_weights2.*repmat(reshape(back1,1,[]),hidden1_num_neuron,1), [], batch_size),2);
                                gradient_weights2 = back1 * temp_gradient_weights2;
                                gradient_bias2 = sum(back1,2);%backward_bias(output_num_neuron,batch_size)*reshape(back1,[],1);       
                                back2 = backward(weights2)*back1;
                                %back3 = backward_act(out1,back2);
                                back3 = backward_act(out1).*back2;
                                %temp_gradient_weights1 = backward_weight_faster(input,input_num_neuron,hidden1_num_neuron);
                                temp_gradient_weights1 = backward_weight_slides(input);
                                %gradient_weights1 = temp_gradient_weights1*reshape(back3,[],1);
                                %gradient_weights1 = sum(reshape(temp_gradient_weights1.*repmat(reshape(back3,1,[]),input_num_neuron,1), [], batch_size),2);
                                gradient_weights1 = back3 * temp_gradient_weights1;
                                gradient_bias1 = sum(back3,2);%backward_bias(hidden1_num_neuron,batch_size)*reshape(back3,[],1);

                                %weight_update

                %                 new_gradient_weights1 = momentum * old_gradient_weights1 - lr * (reshape(gradient_weights1,[],hidden1_num_neuron))'/batch_size;
                %                 new_gradient_weights2 = momentum * old_gradient_weights2 - lr * (reshape(gradient_weights2,[],output_num_neuron))'/batch_size;
                               new_gradient_weights1 = momentum * old_gradient_weights1 - lr * gradient_weights1/batch_size;
                                new_gradient_weights2 = momentum * old_gradient_weights2 - lr * gradient_weights2/batch_size;

                                weights1 = weights1 - w_decay*lr*weights1 + new_gradient_weights1;
                                weights2 = weights2 - w_decay*lr*weights2  + new_gradient_weights2;
                                bias1 = bias1 - lr*gradient_bias1/batch_size;
                                bias2 = bias2 - lr*gradient_bias2/batch_size;

                                old_gradient_weights1 = new_gradient_weights1;
                                old_gradient_weights2 = new_gradient_weights2;

                                %disp(' ');

                            end
                            %disp([num2str(epoch), ' finished']);
                %             disp('Training set loss');
                %             disp((train_loss/3000));
                %             disp('Training set error');
                %             disp((train_error/3000));

                            %validation
                            out1 = forward(valid_data', weights1, bias1);
                            out1_act = forward_act(out1);
                            out1_act = out1_act * (1-dropout_prob);
                            out2 = forward(out1_act,weights2, bias2);
                            final_out = forward_softmax(out2);
                            final_error_num = errorNum(final_out,(valid_label_single+1)');
                            final_loss = loss_func(final_out, valid_label');
                            valid_loss(epoch) = valid_loss(epoch) + sum(final_loss);
                            valid_error(epoch) = valid_error(epoch) + final_error_num;
                %             disp('validation set loss');
                %             disp((valid_loss/1000));
                %             disp('validation set misclassification rate');
                %             disp((valid_error/1000));

                        %test
                    %     for batch=1:batch_num
                    %         out1 = forward(valid_data', weights1, bias1);
                    %         out1_act = forward_act(out1);
                    %         out2 = forward(out1_act,weights2, bias2);
                    %         final_out = forward_softmax(out2);
                    %         final_loss = loss_func(final_out, valid_label');
                    %         valid_loss(epoch) = valid_loss(epoch) + sum(final_loss);
                    %         disp('validation set loss');
                    %         disp((valid_loss/1000));
                    %     end
                        end
                        final_error_rate = valid_error(epoch_num)/1000;
                        disp(final_error_rate);
                        %save(['Problem_f_lr',num2str(lr),'_','momentum',num2str(momentum),'_','numneuron',num2str(hidden1_num_neuron),'-2kEpoch-dropout.mat']);
                        save(['MNIST_2out_',num2str(lr),'_',num2str(momentum),'_',num2str(hidden1_num_neuron),'_',num2str(epoch_num),'_',num2str(w_decay),'_',num2str(dropout_prob),'.mat']);
%                     end
%                 end
%             end
%         end
%     end
%     
% end