clc;
clear;

%% for plotting prob a,b
% myFontSize = 10;
% figure();
% for i=1:10
%     load(['Problem_ab_run', num2str(i), '.mat']);
%     subplot(2,5,i)
%     plot([1:200], train_loss/3000, 'b');
%     hold on;
%     plot([1:200], valid_loss/1000, 'r');
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Cross-entropy','FontSize', myFontSize+5);
%     title(['Run ', num2str(i), ': cross-entropy']);
%     h_legend = legend('Training', 'validation');
%     set(h_legend,'FontSize',myFontSize+5);
% end
% 
% figure();
% for i=1:10
%     load(['Problem_ab_run', num2str(i), '.mat']);
%     subplot(2,5,i)
%     plot([1:200], train_error/3000, 'b');
%     hold on;
%     plot([1:200], valid_error/1000, 'r');
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Misclassification error rate','FontSize', myFontSize+5);
%     title(['Run ', num2str(i), ': misclassification error'])
%     h_legend = legend('Training', 'validation');
%     set(h_legend,'FontSize',myFontSize+5);
% end

%% run1 gives the lowest error rate of 7.3%
% for i=1:10
%     load(['Problem_ab_run', num2str(i), '.mat']);
%     disp(min(valid_error/1000));
% end

%% plot 100 28*28 weights
% load('Problem_ab_run1.mat');
% figure()
% min_weight = min(min(weights1));
% weights1_offset = weights1 - min_weight;
% max_weight = max(max(weights1_offset));
% for i=1:100
%     h = subplot(10,10,i);
%     imshow(reshape(weights1_offset(i,:),28,28)'/max_weight);
%     p = get(h,'pos');
%     p(3) = p(3) + 0.01;
%     p(4) = p(4) + 0.01;
%     set(h,'pos',p);
%     axis('off');
% end

%% for plotting prob d
% myFontSize = 10;
% figure();
% lr=0.1;
% momentum = 0.9;
% i=0;
% for lr=[0.1,0.01,0.2,0.5]
%     i=i+1;
%     load(['Problem_d_lr',num2str(lr),'_','momentum',num2str(momentum),'.mat']);
%     subplot(4,1,i)
%     plot([1:200], train_loss/3000, 'b');
%     hold on;
%     plot([1:200], valid_loss/1000, 'r');
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Cross-entropy','FontSize', myFontSize+5);
%     title(['lr:',num2str(lr),'-','momentum:',num2str(momentum)]);
%     h_legend = legend('Training', 'validation');
%     set(h_legend,'FontSize',myFontSize+5);
% end
% 
% figure();
% i=0;
% for lr=[0.1,0.01,0.2,0.5]
%     i=i+1;
%     load(['Problem_d_lr',num2str(lr),'_','momentum',num2str(momentum),'.mat']);
%     subplot(4,1,i)
%     plot([1:200], train_error/3000, 'b');
%     hold on;
%     plot([1:200], valid_error/1000, 'r');
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Classification error rate','FontSize', myFontSize+5);
%     title(['lr:',num2str(lr),'-','momentum:',num2str(momentum)]);
%     h_legend = legend('Training', 'validation');
%     set(h_legend,'FontSize',myFontSize+5);
% end
% 
% lr=0.1;
% momentum = 0.9;
% figure();
% i=0;
% for momentum=[0.0,0.5, 0.9]
%     i=i+1;
%     load(['Problem_d_lr',num2str(lr),'_','momentum',num2str(momentum),'.mat']);
%     subplot(3,1,i)
%     plot([1:200], train_loss/3000, 'b');
%     hold on;
%     plot([1:200], valid_loss/1000, 'r');
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Cross-entropy','FontSize', myFontSize+5);
%     title(['lr:',num2str(lr),'-','momentum:',num2str(momentum)]);
%     h_legend = legend('Training', 'validation');
%     set(h_legend,'FontSize',myFontSize+5);
% end
% 
% figure();
% i=0;
% for momentum=[0.0,0.5, 0.9]
%     i=i+1;
%     load(['Problem_d_lr',num2str(lr),'_','momentum',num2str(momentum),'.mat']);
%     subplot(3,1,i)
%     plot([1:200], train_error/3000, 'b');
%     hold on;
%     plot([1:200], valid_error/1000, 'r');
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Classification error rate','FontSize', myFontSize+5);
%     title(['lr:',num2str(lr),'-','momentum:',num2str(momentum)]);
%     h_legend = legend('Training', 'validation');
%     set(h_legend,'FontSize',myFontSize+5);
% end

% %% for plotting prob d
% myFontSize = 10;
% figure();
% lr=0.1;
% momentum = 0.9;
% i=0;
% subplot(2,1,1)
% for lr=[0.1,0.01,0.2,0.5]
%     i=i+1;
%     load(['Problem_d_lr',num2str(lr),'_','momentum',num2str(momentum),'.mat']);
%     
%     plot([1:200], train_loss/3000);
%     hold on;
%     
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Cross-entropy','FontSize', myFontSize+5);
%     title(['momentum:',num2str(momentum),'-training']);
%     h_legend = legend('lr=0.1', 'lr=0.01','lr=0.2','lr=0.5');
%     set(h_legend,'FontSize',myFontSize+5);
% end
% subplot(2,1,2)
% for lr=[0.1,0.01,0.2,0.5]
%     i=i+1;
%     load(['Problem_d_lr',num2str(lr),'_','momentum',num2str(momentum),'.mat']);
%     
%     plot([1:200], valid_loss/1000);
%     hold on;
%     
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Cross-entropy','FontSize', myFontSize+5);
%     title(['momentum:',num2str(momentum),'-validation']);
%     h_legend = legend('lr=0.1', 'lr=0.01','lr=0.2','lr=0.5');
%     set(h_legend,'FontSize',myFontSize+5);
% end
% figure();
% lr=0.1;
% momentum = 0.9;
% i=0;
% subplot(2,1,1)
% for lr=[0.1,0.01,0.2,0.5]
%     i=i+1;
%     load(['Problem_d_lr',num2str(lr),'_','momentum',num2str(momentum),'.mat']);
%     
%     plot([1:200], train_error/3000);
%     hold on;
%     
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Classification error rate','FontSize', myFontSize+5);
%     title(['momentum:',num2str(momentum),'-training']);
%     h_legend = legend('lr=0.1', 'lr=0.01','lr=0.2','lr=0.5');
%     set(h_legend,'FontSize',myFontSize+5);
% end
% subplot(2,1,2)
% for lr=[0.1,0.01,0.2,0.5]
%     i=i+1;
%     load(['Problem_d_lr',num2str(lr),'_','momentum',num2str(momentum),'.mat']);
%     
%     plot([1:200], valid_error/1000);
%     hold on;
%     
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Classification error rate','FontSize', myFontSize+5);
%     title(['momentum:',num2str(momentum),'-validation']);
%     h_legend = legend('lr=0.1', 'lr=0.01','lr=0.2','lr=0.5');
%     set(h_legend,'FontSize',myFontSize+5);
% end


%% for plotting prob e
% myFontSize = 10;
% figure();
% lr=0.01;
% momentum = 0.5;
% hidden_neuron = 100;
% i=0;
% for hidden_neuron=[20,100,200,500]
%     i=i+1;
%     load(['Problem_e_lr',num2str(lr),'_','momentum',num2str(momentum),'_','numneuron',num2str(hidden_neuron),'-2kEpoch.mat']);
%     subplot(2,2,i)
%     plot([1:2000], train_loss/3000, 'b');
%     hold on;
%     plot([1:2000], valid_loss/1000, 'r');
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Cross-entropy','FontSize', myFontSize+5);
%     title(['lr:',num2str(lr),'-','momentum:',num2str(momentum),'-','hiddenNeuron:',num2str(hidden_neuron)]);
%     h_legend = legend('Training', 'validation');
%     set(h_legend,'FontSize',myFontSize+5);
% end
% 
% figure();
% i=0;
% for hidden_neuron=[20,100,200,500]
%     i=i+1;
%     load(['Problem_e_lr',num2str(lr),'_','momentum',num2str(momentum),'_','numneuron',num2str(hidden_neuron),'-2kEpoch.mat']);
%     subplot(2,2,i)
%     plot([1:2000], train_error/3000, 'b');
%     hold on;
%     plot([1:2000], valid_error/1000, 'r');
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Classification error rate','FontSize', myFontSize+5);
%     title(['lr:',num2str(lr),'-','momentum:',num2str(momentum),'-','hiddenNeuron:',num2str(hidden_neuron)]);
%     h_legend = legend('Training', 'validation');
%     set(h_legend,'FontSize',myFontSize+5);
% end

%% for plotting prob e
% myFontSize = 10;
% figure();
% lr=0.01;
% momentum = 0.5;
% i=0;
% subplot(2,1,1)
% for  hidden_neuron=[20,100,200,500]
%     i=i+1;
%     load(['Problem_e_lr',num2str(lr),'_','momentum',num2str(momentum),'_','numneuron',num2str(hidden_neuron),'-2kEpoch.mat']);
%     
%     plot([1:2000], train_loss/3000);
%     hold on;
%     
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Cross-entropy','FontSize', myFontSize+5);
%     title(['lr:',num2str(lr),'-','momentum:',num2str(momentum),'-training']);
%     h_legend = legend('#hiddenNeuron=20', '#hiddenNeuron=100','#hiddenNeuron=200','#hiddenNeuron=500');
%     set(h_legend,'FontSize',myFontSize+5);
% end
% subplot(2,1,2)
% for  hidden_neuron=[20,100,200,500]
%     i=i+1;
%     load(['Problem_e_lr',num2str(lr),'_','momentum',num2str(momentum),'_','numneuron',num2str(hidden_neuron),'-2kEpoch.mat']);
%     
%     plot([1:2000], valid_loss/1000);
%     hold on;
%     
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Cross-entropy','FontSize', myFontSize+5);
%     title(['lr:',num2str(lr),'-','momentum:',num2str(momentum),'-validation']);
%     h_legend = legend('#hiddenNeuron=20', '#hiddenNeuron=100','#hiddenNeuron=200','#hiddenNeuron=500');
%     set(h_legend,'FontSize',myFontSize+5);
% end
% figure();
% i=0;
% subplot(2,1,1)
% for hidden_neuron=[20,100,200,500]
%     i=i+1;
%     load(['Problem_e_lr',num2str(lr),'_','momentum',num2str(momentum),'_','numneuron',num2str(hidden_neuron),'-2kEpoch.mat']);
%     plot([1:2000], train_error/3000);
%     hold on;
%     
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Classification error rate','FontSize', myFontSize+5);
%     title(['lr:',num2str(lr),'-','momentum:',num2str(momentum),'-training']);
%     h_legend = legend('#hiddenNeuron=20', '#hiddenNeuron=100','#hiddenNeuron=200','#hiddenNeuron=500');
%     set(h_legend,'FontSize',myFontSize+5);
% end
% subplot(2,1,2)
% for hidden_neuron=[20,100,200,500]
%     i=i+1;
%     load(['Problem_e_lr',num2str(lr),'_','momentum',num2str(momentum),'_','numneuron',num2str(hidden_neuron),'-2kEpoch.mat']);
%     
%     plot([1:2000], valid_error/1000);
%     hold on;
%     disp(valid_error(2000)/1000);
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Classification error rate','FontSize', myFontSize+5);
%     title(['lr:',num2str(lr),'-','momentum:',num2str(momentum),'-validation']);
%     h_legend = legend('#hiddenNeuron=20', '#hiddenNeuron=100','#hiddenNeuron=200','#hiddenNeuron=500');
%     set(h_legend,'FontSize',myFontSize+5);
% end

%% for plotting prob d
% myFontSize = 10;
% figure();
% lr=0.1;
% momentum = 0.9;
% i=0;
% subplot(2,1,1)
% for  momentum=[0.0,0.5, 0.9]
%     i=i+1;
%     load(['Problem_d_lr',num2str(lr),'_','momentum',num2str(momentum),'.mat']);
%     
%     plot([1:200], train_loss/3000);
%     hold on;
%     
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Cross-entropy','FontSize', myFontSize+5);
%     title(['lr:',num2str(lr),'-training']);
%     h_legend = legend('momentum=0.0', 'momentum=0.5','momentum=0.9');
%     set(h_legend,'FontSize',myFontSize+5);
% end
% subplot(2,1,2)
% for  momentum=[0.0,0.5, 0.9]
%     i=i+1;
%     load(['Problem_d_lr',num2str(lr),'_','momentum',num2str(momentum),'.mat']);
%     
%     plot([1:200], valid_loss/1000);
%     hold on;
%     
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Cross-entropy','FontSize', myFontSize+5);
%     title(['lr:',num2str(lr),'-validation']);
%     h_legend = legend('momentum=0.0', 'momentum=0.5','momentum=0.9');
%     set(h_legend,'FontSize',myFontSize+5);
% end
% figure();
% lr=0.1;
% momentum = 0.9;
% i=0;
% subplot(2,1,1)
% for momentum=[0.0,0.5, 0.9]
%     i=i+1;
%     load(['Problem_d_lr',num2str(lr),'_','momentum',num2str(momentum),'.mat']);
%     
%     plot([1:200], train_error/3000);
%     hold on;
%     
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Classification error rate','FontSize', myFontSize+5);
%     title(['lr:',num2str(lr),'-training']);
%     h_legend = legend('momentum=0.0', 'momentum=0.5','momentum=0.9');
%     set(h_legend,'FontSize',myFontSize+5);
% end
% subplot(2,1,2)
% for momentum=[0.0,0.5, 0.9]
%     i=i+1;
%     load(['Problem_d_lr',num2str(lr),'_','momentum',num2str(momentum),'.mat']);
%     
%     plot([1:200], valid_error/1000);
%     hold on;
%     
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Classification error rate','FontSize', myFontSize+5);
%     title(['lr:',num2str(lr),'-validation']);
%     h_legend = legend('momentum=0.0', 'momentum=0.5','momentum=0.9');
%     set(h_legend,'FontSize',myFontSize+5);
% end

%% for plotting prob f
% myFontSize = 10;
% figure();
% lr=0.01;
% momentum = 0.5;
% hidden_neuron = 100;
% i=0;
% for hidden_neuron=[100,200,500]
%     i=i+1;
%     load(['Problem_f_lr',num2str(lr),'_','momentum',num2str(momentum),'_','numneuron',num2str(hidden_neuron),'-2kEpoch-dropout.mat']);
%     subplot(3,1,i)
%     plot([1:2000], train_loss/3000, 'b');
%     hold on;
%     plot([1:2000], valid_loss/1000, 'r');
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Cross-entropy','FontSize', myFontSize+5);
%     title(['lr:',num2str(lr),'-','momentum:',num2str(momentum),'-','hiddenNeuron:',num2str(hidden_neuron)]);
%     h_legend = legend('Training', 'validation');
%     set(h_legend,'FontSize',myFontSize+5);
% end
% 
% figure();
% i=0;
% for hidden_neuron=[100,200,500]
%     i=i+1;
%     load(['Problem_f_lr',num2str(lr),'_','momentum',num2str(momentum),'_','numneuron',num2str(hidden_neuron),'-2kEpoch-dropout.mat']);
%     subplot(3,1,i)
%     plot([1:2000], train_error/3000, 'b');
%     hold on;
%     plot([1:2000], valid_error/1000, 'r');
%     disp(valid_error(2000)/1000);
%     xlabel('Epoch','FontSize', myFontSize+5);
%     ylabel('Classification error rate','FontSize', myFontSize+5);
%     title(['lr:',num2str(lr),'-','momentum:',num2str(momentum),'-','hiddenNeuron:',num2str(hidden_neuron)]);
%     h_legend = legend('Training', 'validation');
%     set(h_legend,'FontSize',myFontSize+5);
% end

%% plot 100 28*28 weights
%load('Problem_g_0.1_0.9_200_600_0.001_0.5.mat')
load('Problem_h_0.1_0.9_200_200_600_0.001_0.5-twohidden.mat')
figure()
min_weight = min(min(weights1));
weights1_offset = weights1 - min_weight;
max_weight = max(max(weights1_offset));
for i=1:100
    h = subplot(10,10,i);
    imshow(reshape(weights1_offset(i,:),28,28)'/max_weight);
    p = get(h,'pos');
    p(3) = p(3) + 0.01;
    p(4) = p(4) + 0.01;
    set(h,'pos',p);
    axis('off');
end