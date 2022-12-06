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
selected_users_IDs = double(selected_users_IDs);  % ���ݸ�ʽת��
offset1 = max(selected_checkins(:,1));
[~,~,n] = unique(selected_checkins(:,2));  % n:Ԫ���ڽ��غ������λ��
selected_checkins(:,2) = n+offset1;

offset2 = max(selected_checkins(:,2));
[~,~,n] = unique(selected_checkins(:,3));
selected_checkins(:,3) = n+offset2;

offset3 = max(selected_checkins(:,3));
[~,~,n] = unique(selected_checkins(:,4));
selected_checkins(:,4) = n+offset3;

num_node_total = max(selected_checkins(:));

%%% 2. prepare checkins per user (fast)
user_checkins = cell(length(selected_users_IDs),1);  % ������Ԫ�飨�û�����1��
temp_checkins = sortrows(selected_checkins,1);  % ���û�id������
[u,m,~] = unique(temp_checkins(:,1));  % u:���غ�ľ���m:Ԫ����ԭ�����е�λ��
counters = [m(2:end);size(temp_checkins,1)+1] - m; % �û�ǩ��Ƶ����
user_checkins(u) = mat2cell(temp_checkins,counters,4);  % ����ֿ�
user_checkins = cellfun(@transpose,user_checkins,'UniformOutput',false); % ��ÿ���Ӿ���ת��
user_checkins = cellfun(@int64,user_checkins,'UniformOutput',false);

user_checkins_counter = zeros(length(selected_users_IDs),1,'int64');
user_checkins_counter(u) = int64(counters);

%%% 3. random walk
num_node = length(selected_users_IDs);
network = sparse(friendship_old(:,1), friendship_old(:,2),ones(size(friendship_old,1),1),num_node, num_node);% �����罻�������
network = network+network';  % �罻������������һ��(˫��

node_list = cell(num_node,1);
node_list_len = zeros(num_node,1);
num_walk = 10;
len_walk = 80;
[indy,indx] = find(network');
[temp,m,n] = unique(indx);
node_list_len(temp) = [m(2:end);length(indx)+1] - m; % sum(counts)
node_list(temp) = mat2cell(indy,node_list_len(temp));  % �罻���簴�зֿ顣 ÿһ��Ϊһ���û������к��ѹ�ϵ

% find(node_list_len==0) % should be empty, otherwise there exists isolated nodes !!!

% let's have a walk over social network (friendship)
walks = zeros(num_walk*num_node,len_walk,'int64');  % �洢�������������ݾ���
for ww=1:num_walk % 10
    for ii=1:num_node % 4024
        % ����ÿһ���û���len_walk���罻�� 
        seq = zeros(1,len_walk); % 80
        seq(1) = ii;
        current_e = ii;
        
        if node_list_len(ii) == 0
            walks(ii+(ww-1)*num_node,:) = seq;  % ���罻��ϵ������ѭ��
            continue
        end
        
        for jj=1:len_walk-1
            rand_ind = randi([1 node_list_len(seq(jj))],1);
            seq(jj+1) = node_list{seq(jj)}(rand_ind,:);
        end
        walks(ii+(ww-1)*num_node,:) = seq;
        % д��һ�У�һ������Ϊ���û�1-�û�1������a-�û�1������a������b-...-len_walk����
    end
end

%%% 4. ������
% social relationship
[r,~] = find(network);
tab_degree = tabulate(r); % ����: ֵ Ƶ�� Ƶ��
freq = tab_degree(:,3).^(0.75); 
% 0.75�η� Ϊ����ֵ...
% ��һ�֡�ƽ�������ԣ��ܹ��õ�Ƶ�ʶ�һЩ�������ᣬ��Ƶ�ʹ���һЩ��������
neg_sam_table_social = int64(repelem(tab_degree(:,1),round(1000000* freq/sum(freq)))); % unigram with 0.75 power
% repelem: �ظ�����Ԫ�ظ���
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

