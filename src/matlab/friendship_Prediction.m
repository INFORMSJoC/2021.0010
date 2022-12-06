close all ; clear ; clc;
tic;
load('embs.mat');
load('LBSN2vec_input.mat');

num_sample_evaluate = 0.1; % 最后评估时对用户节点对随机取样百分数
num_users = size(embs_user, 1);  % 用户数量
num_pre_rank = 0.9;  % K
pairs_node_user_mat = zeros(num_users, num_users);  % 创建用户节点对矩阵（对称阵）
% 数据格式转换A
friendship_old = double(friendship_old);
selected_checkins = double(selected_checkins);
selected_users_IDs = double(selected_users_IDs); 
%% 社交关系
% 旧社交关系
network = sparse(friendship_old(:,1), friendship_old(:,2),ones(size(friendship_old,1),1),num_users, num_users);% 单向社交网络矩阵
network = network+network';  % 社交网络行列连接一致(双向）
node_list_old = cell(num_users,1);
node_list_len = zeros(num_users,1);
[indy,indx] = find(network');
[temp,m,~] = unique(indx);
node_list_len(temp) = [m(2:end);length(indx)+1] - m; % sum(counts)
node_list_old(temp) = mat2cell(indy,node_list_len(temp));


for i = 1:num_users
   for j = setdiff(1:num_users, node_list_old{i})  % 去除旧社交关系中的好友
       pairs_node_user_mat(i, j) = 1;   
   end
end
tri = triu(pairs_node_user_mat,1);  % 上三角矩阵中值为1的坐标为一对用户节点
[x, y] = find(tri == 1);  % 得到所有用户节点对
pairs_node = [x, y]; 
[m,~] = size(pairs_node); % 用户节点对数量
rand_sam = randperm(m, ceil(m*num_sample_evaluate)); % 取样随机数坐标
cosine_sim_user = zeros(length(rand_sam), 3);  % 用户节点对余弦相似度,用户对id

for i = 1:length(rand_sam)
    row1 = pairs_node(rand_sam(i), 1);  % 用户1
    cosine_sim_user(i, 2) = row1;
    row2 = pairs_node(rand_sam(i), 2);  % 用户2
    cosine_sim_user(i, 3) = row2;
    %%% 余弦相似度计算
    cosine_sim_user(i, 1) = sum(embs_user(row1, :) .* embs_user(row2, :)) ...
    / (sqrt(sum(embs_user(row1, :).^2)) * sqrt(sum(embs_user(row2, :).^2)));
end

sorted_cosine_sim_user = sortrows(cosine_sim_user, -1); % 排序
% predict_users_rank = sorted_cosine_sim_user(1:num_pre_rank,:);
predict_users_rank = sorted_cosine_sim_user(sorted_cosine_sim_user(:,1)>num_pre_rank,:);
save('predict_users_rank.mat','predict_users_rank','-v7.3');

toc;