close all ; clear ; clc;
%% load prepared dataset
load('LBSN2vec_input.mat');
% data format:
% selected_checkins (4 columns): user_index, time (hour in a week), venue_index, venue_category_index
% friendship_old/friendship_new (2 columns): user_index, user_index
% selected_user_IDs(user_index) => anonymized user IDs
% selected_venue_IDs(venue_index) => venue IDs
%% 

%% preprocessing
%%% 1. rebuild node index
friendship_old = double(friendship_old);
selected_checkins = double(selected_checkins);
selected_users_IDs = double(selected_users_IDs);  % 数据格式转换
offset1 = max(selected_checkins(:,1));
[~,~,n] = unique(selected_checkins(:,2));  % n:元素在降重后矩阵中位置
selected_checkins(:,2) = n+offset1;

offset2 = max(selected_checkins(:,2));
[~,~,n] = unique(selected_checkins(:,3));
selected_checkins(:,3) = n+offset2;

offset3 = max(selected_checkins(:,3));
[~,~,n] = unique(selected_checkins(:,4));
selected_checkins(:,4) = n+offset3;

num_node_total = max(selected_checkins(:));

%%% 2. prepare checkins per user (fast)
user_checkins = cell(length(selected_users_IDs),1);  % 创建空元组（用户数，1）
temp_checkins = sortrows(selected_checkins,1);  % 按用户id列排序
[u,m,~] = unique(temp_checkins(:,1));  % u:降重后的矩阵，m:元素在原矩阵中的位置
counters = [m(2:end);size(temp_checkins,1)+1] - m; % 用户签到频数？
user_checkins(u) = mat2cell(temp_checkins,counters,4);  % 矩阵分块
user_checkins = cellfun(@transpose,user_checkins,'UniformOutput',false); % 对每个子矩阵转置
user_checkins = cellfun(@int64,user_checkins,'UniformOutput',false);

user_checkins_counter = zeros(length(selected_users_IDs),1,'int64');
user_checkins_counter(u) = int64(counters);

%%% 3. random walk
num_node = length(selected_users_IDs);
network = sparse(friendship_old(:,1), friendship_old(:,2),ones(size(friendship_old,1),1),num_node, num_node);% 单向社交网络矩阵
network = network+network';  % 社交网络行列连接一致(双向）

node_list = cell(num_node,1);
node_list_len = zeros(num_node,1);
num_walk = 10;
len_walk = 80;
[indy,indx] = find(network');
[temp,m,n] = unique(indx);
node_list_len(temp) = [m(2:end);length(indx)+1] - m; % sum(counts)
node_list(temp) = mat2cell(indy,node_list_len(temp));  % 社交网络按行分块。 每一行为一个用户的所有好友关系

% find(node_list_len==0) % should be empty, otherwise there exists isolated nodes !!!

% let's have a walk over social network (friendship)
walks = zeros(num_walk*num_node,len_walk,'int64');  % 存储随机漫步后的数据矩阵
for ww=1:num_walk % 10
    for ii=1:num_node % 4024
        % 遍历每一个用户的len_walk个社交链 
        seq = zeros(1,len_walk); % 80
        seq(1) = ii;
        current_e = ii;
        
        if node_list_len(ii) == 0
            walks(ii+(ww-1)*num_node,:) = seq;  % 无社交关系则跳出循环
            continue
        end
        
        for jj=1:len_walk-1
            rand_ind = randi([1 node_list_len(seq(jj))],1);
            seq(jj+1) = node_list{seq(jj)}(rand_ind,:);
        end
        walks(ii+(ww-1)*num_node,:) = seq;
        % 写入一行，一行数据为：用户1-用户1的朋友a-用户1的朋友a的朋友b-...-len_walk条。
    end
end

%%% 4. 负采样
% social relationship
[r,~] = find(network);
tab_degree = tabulate(r); % 返回: 值 频数 频率
freq = tab_degree(:,3).^(0.75); 
% 0.75次方 为经验值...
% 是一种“平滑”策略，能够让低频词多一些出场机会，高频词贡献一些出场机会
neg_sam_table_social = int64(repelem(tab_degree(:,1),round(1000000* freq/sum(freq)))); % unigram with 0.75 power
% repelem: 重复数组元素副本
% checkins: user, venue, time, semantic, with node type normalization for each node do0main
neg_sam_table_mobility_norm = cell(4,1);
for ii=1:length(neg_sam_table_mobility_norm)
    tab_degree = tabulate(temp_checkins(:,ii)); 
    freq = tab_degree(:,3).^(0.75);
    u = round(100000* freq/sum(freq));
    neg_sam_table_mobility_norm{ii} = int64(repelem(tab_degree(:,1),round(100000* freq/sum(freq)))); % unigram with 0.75 power
end

% clean a bit the variables
clearvars -except friendship_new friendship_old selected_checkins selected_users_IDs selected_venue...
    num_node_total offset1 offset2 offset3...
    walks user_checkins user_checkins_counter neg_sam_table_social neg_sam_table_mobility_norm node_list;




%% LBSN2vec
dim_emb = 128;
num_epoch = 1;
num_threads =  4;
K_neg = 10;
win_size = 10;
learning_rate = 0.001;

embs_ini = (rand(num_node_total,dim_emb)-0.5)/dim_emb; %
embs_len = sqrt(sum(embs_ini.^(2), 2));
embs_ini = embs_ini./(repmat(embs_len, 1, dim_emb));

mobility_ratio = 0.2; % how much mobility influence is considered (how many checkins are learnt for one node in walks)

tic;
[embs] = learn_LBSN2Vec_embedding(walks',user_checkins, user_checkins_counter,...
    embs_ini', learning_rate, K_neg,...
    neg_sam_table_social, win_size, neg_sam_table_mobility_norm,...
    num_epoch,num_threads,mobility_ratio);
toc;
embs = embs';
embs_len = sqrt(sum(embs.^(2), 2));
embs = embs./(repmat(embs_len, 1, dim_emb));

% get node embeddings in individual domains
embs_user = embs(1:offset1,:);
embs_time = embs(offset1+1:offset2,:);
embs_venue = embs(offset2+1:offset3,:);
embs_cate = embs(offset3+1:end,:);

save('embs.mat','embs','embs_user','embs_time','embs_venue','embs_cate');

