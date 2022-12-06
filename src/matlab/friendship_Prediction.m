close all ; clear ; clc;
tic;
load('embs.mat');
load('LBSN2vec_input.mat');

num_sample_evaluate = 0.1; % �������ʱ���û��ڵ�����ȡ���ٷ���
num_users = size(embs_user, 1);  % �û�����
num_pre_rank = 0.9;  % K
pairs_node_user_mat = zeros(num_users, num_users);  % �����û��ڵ�Ծ��󣨶Գ���
% ���ݸ�ʽת��A
friendship_old = double(friendship_old);
selected_checkins = double(selected_checkins);
selected_users_IDs = double(selected_users_IDs); 
%% �罻��ϵ
% ���罻��ϵ
network = sparse(friendship_old(:,1), friendship_old(:,2),ones(size(friendship_old,1),1),num_users, num_users);% �����罻�������
network = network+network';  % �罻������������һ��(˫��
node_list_old = cell(num_users,1);
node_list_len = zeros(num_users,1);
[indy,indx] = find(network');
[temp,m,~] = unique(indx);
node_list_len(temp) = [m(2:end);length(indx)+1] - m; % sum(counts)
node_list_old(temp) = mat2cell(indy,node_list_len(temp));


for i = 1:num_users
   for j = setdiff(1:num_users, node_list_old{i})  % ȥ�����罻��ϵ�еĺ���
       pairs_node_user_mat(i, j) = 1;   
   end
end
tri = triu(pairs_node_user_mat,1);  % �����Ǿ�����ֵΪ1������Ϊһ���û��ڵ�
[x, y] = find(tri == 1);  % �õ������û��ڵ��
pairs_node = [x, y]; 
[m,~] = size(pairs_node); % �û��ڵ������
rand_sam = randperm(m, ceil(m*num_sample_evaluate)); % ȡ�����������
cosine_sim_user = zeros(length(rand_sam), 3);  % �û��ڵ���������ƶ�,�û���id

for i = 1:length(rand_sam)
    row1 = pairs_node(rand_sam(i), 1);  % �û�1
    cosine_sim_user(i, 2) = row1;
    row2 = pairs_node(rand_sam(i), 2);  % �û�2
    cosine_sim_user(i, 3) = row2;
    %%% �������ƶȼ���
    cosine_sim_user(i, 1) = sum(embs_user(row1, :) .* embs_user(row2, :)) ...
    / (sqrt(sum(embs_user(row1, :).^2)) * sqrt(sum(embs_user(row2, :).^2)));
end

sorted_cosine_sim_user = sortrows(cosine_sim_user, -1); % ����
% predict_users_rank = sorted_cosine_sim_user(1:num_pre_rank,:);
predict_users_rank = sorted_cosine_sim_user(sorted_cosine_sim_user(:,1)>num_pre_rank,:);
save('predict_users_rank.mat','predict_users_rank','-v7.3');

toc;