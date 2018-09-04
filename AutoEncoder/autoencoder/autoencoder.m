clc;
clear; 
temp_train_set = csvread('../digitstrain.txt');
valid_set = csvread('../digitsvalid.txt');
test_set = csvread('../digitstest.txt');

%%
 train_set = temp_train_set(randperm(size(temp_train_set,1)),:);
train_data = train_set(:,1:784);
train_data = train_data>0.5;

train_label_single = train_set(:,785);
train_label = zeros(3000,10);
for iter=1:3000
    train_label(iter,train_label_single(iter)+1) = 1;
end

valid_data = valid_set(:,1:784);
test_data = test_set(:,1:784);
valid_data = valid_data > 0.5;
test_data = test_data>0.5;

valid_label_single = valid_set(:,785);
valid_label = zeros(1000,10);
for iter=1:1000
    valid_label(iter,valid_label_single(iter)+1) = 1;
end
test_label = test_set(:,785);

input_num_neuron = 784;
hidden1_num_neuron = 100;
output_num_neuron = 784;

lr=0.01;
momentum=0.0;
w_decay = 0.0; %0.0005;
flip_prob = 0.0; %0.5;
epoch_num = 50;
batch_size = 1;
batch_num = 3000/batch_size;


%for runs=5:10
% for lr=[0.01,0.1]
%     for momentum=[0.5, 0.9]
        for flip_prob=[0,0.25]

         for hidden1_num_neuron=[50,100,200,500]
%             for epoch_num=[200, 600]
%                 for w_decay=[0.0005, 0.001]
%                     for dropout_prob=[0.5, 0.9]
                        %disp(['HW2_Problem_e_sqrloss_',num2str(lr),'_',num2str(hidden1_num_neuron),'_',num2str(epoch_num),'_',num2str(batch_size),'_',num2str(dropout_prob)]);

                        %initialize weights
                        rand_bound1 = sqrt(6)/sqrt(hidden1_num_neuron+input_num_neuron);

                        rand_weights1 = -rand_bound1 + 2*rand_bound1*rand(hidden1_num_neuron, input_num_neuron);
                        
                        weights = rand_weights1;
                        bias1 = zeros(hidden1_num_neuron,1);
                        bias2 = zeros(output_num_neuron,1);

                        train_loss = zeros(1,epoch_num);
                        valid_loss = zeros(1,epoch_num);
                        for epoch=1:epoch_num  
%                             rand_shuffle = randperm(size(temp_train_set,1));
%                             train_data = train_data(rand_shuffle,:);
%                             train_label = train_label(rand_shuffle,:);
                            %disp(['Epoch: ', num2str(epoch)]);
                            for batch=1:batch_num
                                input = train_data((batch-1)*batch_size+1:batch*batch_size,:)';
                                label = train_label((batch-1)*batch_size+1:batch*batch_size,:)';
                                %forward
                                [corrupt_input,~] = flip_noise(input, flip_prob);

                                out1 = forward(corrupt_input, weights, bias1);               
                                out1_act = forward_act(out1);
                                out2 = forward(out1_act,weights', bias2);
                                final_out = forward_act(out2);
                                positive = input.*final_out;
                                negative = (1-input).*(1-final_out);
                                temp_loss = (-sum(sum(log(positive+(positive==0)) + log(negative+(negative==0)))));
                                if isnan(temp_loss)
                                    disp('NaN=============================================')
                                    disp(batch)
                                    disp([max(out2),max(final_out)]);
                                end
                                train_loss(epoch) = train_loss(epoch) + temp_loss;

                                %disp([num2str(batch), ': ', num2str(mean(final_loss))]);

                                %backward
                                back1 = back_loss_func(final_out, input);
                                temp_gradient_weights2 = backward_weight_slides(out1_act);
                                gradient_weights2 = back1 * temp_gradient_weights2;
                                back2 = backward(weights')*back1;
                                back3 = backward_act(out1).*back2;
                                temp_gradient_weights1 = backward_weight_slides(input);
                                gradient_weights1 = back3 * temp_gradient_weights1;

                                %weight_update

                                weights = weights - lr * (gradient_weights1+gradient_weights2')/batch_size;
  

                            end
                            %disp([num2str(epoch), ' finished']);
                            disp('Training set loss');
                            disp((train_loss/3000));
                %             disp('Training set error');
                %             disp((train_error/3000));

                            %validation
                            out1 = forward(valid_data', weights, bias1);
                            out1_act = forward_act(out1);
                            out2 = forward(out1_act,weights', bias2);
                            final_out = forward_act(out2);
                            positive = valid_data'.*final_out;
                            negative = (1-valid_data').*(1-final_out);
                            temp_loss = (-sum(sum(log(positive+(positive==0)) + log(negative+(negative==0)))));
                            valid_loss(epoch) = valid_loss(epoch) + temp_loss;
                            disp('validation set loss');
                            disp((valid_loss/1000));
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
                        %save(['HW2_Problem_f_dropout_',num2str(lr),'_',num2str(hidden1_num_neuron),'_',num2str(epoch_num),'_',num2str(batch_size),'_',num2str(flip_prob),'.mat']);
                        save(['HW2_Problem_g_autoencoder',num2str(lr),'_',num2str(hidden1_num_neuron),'_',num2str(epoch_num),'_',num2str(batch_size),'_',num2str(flip_prob),'.mat']);

                        %                     end
%                 end
%             end
         end
     end
%     
% end