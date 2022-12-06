import random
import numpy as np
from tqdm import tqdm 
import tensorflow as tf

if tf.__version__ >= '2.0.0':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    

class DAE(object):
    '''
    定义一个DAE网络类
    '''
    def __init__(self, base_model, DAE_input, learning_rate=0.1, max_epoch=50, batch_size=256, soc_coef=0.01):
        '''
        加载预训练权重训练DAE网络
        args:
            base_model: 预训练模型列表[gtr_rbm, rbm_list, rbm_hid_linear]
            DAE_input:  模型输入列表[input_v, input_c, input_r, weight_g]
        '''
        self.base_model = base_model
        
        self.input_v = DAE_input[0]
        self.input_c = DAE_input[1]
        self.input_r = DAE_input[2]
        self.weight_g = DAE_input[3]
        self.input_v_mask = np.zeros_like(self.input_v)  # 输入mask
        self.input_v_mask[self.input_v != 0] = 1
        
        self.learning_rate = learning_rate
        self.training_epochs = max_epoch
        self.batch_size = batch_size
        self.soc_coef = soc_coef


    def set_weight(self):
        """
        加载预训练的权重
        """
        self.n_condition = len(self.base_model[0].weight_c)  # number of conditional layer
        self.num_RBM = len(self.base_model[1])

        ## 加载DAE初始权重
        self.weights, self.biases= {}, {}
        # encoder gtr_rbm
        lay_n = 1
        gtr_rbm_weight_cv = np.array([self.base_model[0].weight_c[n][0].T for n in range(self.n_condition)])  # 条件层与可见层的连接权重
        gtr_rbm_weight_ch = np.array([self.base_model[0].weight_c[n][1].T for n in range(self.n_condition)])  # 条件层与隐藏层的连接权重
        self.weights.update({f'encoder_w{1}': tf.Variable(self.base_model[0].weight_v.T.astype(np.float32), name=f'w{1}'),
                        f'encoder_w{1}ch': tf.Variable(gtr_rbm_weight_ch.astype(np.float32), name=f'w{1}ch'),})
        self.biases.update({f'encoder_b{1}': tf.Variable(self.base_model[0].bias_h.astype(np.float32), name=f'b{1}')})
        # encoder rbm
        for n in range(self.num_RBM):
            lay_n += 1
            self.weights.update({f'encoder_w{lay_n}': tf.Variable(self.base_model[1][n].weight_v.T.astype(np.float32), name=f'w{lay_n}')})
            self.biases.update({f'encoder_b{lay_n}': tf.Variable(self.base_model[1][n].bias_h.astype(np.float32), name=f'b{lay_n}')})

        # encoder rbm_hid_linear
        lay_n += 1
        self.weights.update({f'encoder_w{lay_n}': tf.Variable(self.base_model[2].weight_v.T.astype(np.float32), name=f'w{lay_n}')})
        self.biases.update({f'encoder_b{lay_n}': tf.Variable(self.base_model[2].bias_h.astype(np.float32), name=f'b{lay_n}')})

        # decoder rbm_hid_linear
        lay_n += 1
        self.weights.update({f'decoder_w{lay_n}': tf.Variable(self.base_model[2].weight_v.astype(np.float32), name=f'w{lay_n}')})
        self.biases.update({f'decoder_b{lay_n}': tf.Variable(self.base_model[2].bias_v.astype(np.float32), name=f'b{lay_n}')})
        # decoder rbm
        for n in range(self.num_RBM-1,-1,-1):
            lay_n += 1
            self.weights.update({f'decoder_w{lay_n}': tf.Variable(self.base_model[1][n].weight_v.astype(np.float32), name=f'w{lay_n}')})
            self.biases.update({f'decoder_b{lay_n}': tf.Variable(self.base_model[1][n].bias_v.astype(np.float32), name=f'b{lay_n}')})

        # decoder gtr_rbm
        lay_n += 1
        self.weights.update({f'decoder_w{lay_n}': tf.Variable(self.base_model[0].weight_v.astype(np.float32), name=f'w{lay_n}'),
                        f'decoder_w{lay_n}cv': tf.Variable(gtr_rbm_weight_cv.astype(np.float32), name=f'w{lay_n}cv'),})
        self.biases.update({f'decoder_b{lay_n}': tf.Variable(self.base_model[0].bias_v.astype(np.float32), name=f'b{lay_n}')})

    
    def DAE(self, v, v_mask, c, weight_g):
        """
        构建DAE模型
        """
        lay_n = 1
        lay_res = {}
        lay_res[lay_n] = tf.nn.relu(tf.matmul(v, self.weights[f'encoder_w{lay_n}']) + 
                                    tf.add_n([tf.matmul(c[n], self.weights[f'encoder_w{lay_n}ch'][n]) for n in range(self.n_condition)]) +
                                    self.biases[f'encoder_b{lay_n}'])
        for n in range(self.num_RBM):
            lay_n += 1
            lay_res[lay_n] = tf.nn.relu(tf.matmul(lay_res[lay_n-1], self.weights[f'encoder_w{lay_n}']) +
                                        self.biases[f'encoder_b{lay_n}'])
        lay_n += 1
        lay_res[lay_n] = tf.matmul(lay_res[lay_n-1],self. weights[f'encoder_w{lay_n}']) + self.biases[f'encoder_b{lay_n}']

        lay_n += 1
        lay_res[lay_n] = tf.nn.relu(tf.matmul(lay_res[lay_n-1], self.weights[f'decoder_w{lay_n}']) + self.biases[f'decoder_b{lay_n}'])

        for n in range(self.num_RBM):
            lay_n += 1
            lay_res[lay_n] = tf.nn.relu(tf.matmul(lay_res[lay_n-1], self.weights[f'decoder_w{lay_n}']) +
                                        self.biases[f'decoder_b{lay_n}'])

        lay_n += 1
        lay_res[lay_n] = (tf.matmul(lay_res[lay_n-1], self.weights[f'decoder_w{lay_n}']) +
                          tf.add_n([tf.matmul(c[n], self.weights[f'decoder_w{lay_n}cv'][n]) for n in range(self.n_condition)]) +
                          tf.matmul(v, weight_g) +
                          self.biases[f'decoder_b{lay_n}']) 
        output_mask = v_mask * lay_res[lay_n]  
        return lay_res[lay_n], output_mask, lay_res[1], lay_res[lay_n-1]

    
    def train_predict(self):
        """
        训练DAE模型
        """
        
        tf.reset_default_graph()
        self.set_weight()
        # 定义输入占位符
        num_venue = self.input_v.shape[1]
        Input_visible = tf.placeholder("float", [None, num_venue], name='Input_visible')
        Input_conditional = tf.placeholder("float", [self.n_condition, None, num_venue], name='Input_conditional')
        Input_regular = tf.placeholder("float", [None, num_venue], name='Input_regular')
        Input_geoWeight = tf.placeholder("float", [num_venue, num_venue], name='Input_geoWeight')
        Input_visible_mask = tf.placeholder("float", [None, num_venue], name='Input_visible_mask')

        # DAE
        Output, Output_mask, fistHid_output, lastHid_output= self.DAE(Input_visible, Input_visible_mask, Input_conditional, Input_geoWeight)

        # 定义代价函数和优化器
        cost = tf.reduce_mean(tf.reduce_sum(tf.square(Input_visible - Output_mask), 1))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cost)

        # 重写梯度，加入社交正则项
        new_grad_var = []
        for grad_var in grads_and_vars:
            if grad_var[1].name == 'w1:0':
                grad = tf.add(grad_var[0], self.soc_coef * tf.matmul(tf.transpose(Input_regular), fistHid_output))
                new_grad_var.append((grad, grad_var[1]))
            elif grad_var[1].name == f'w{(self.num_RBM+2)*2}:0':
                grad = tf.add(grad_var[0], self.soc_coef * (tf.matmul(tf.transpose(lastHid_output), Input_regular)))
                new_grad_var.append((grad, grad_var[1]))
            else:
                new_grad_var.append(grad_var)
        # 应用梯度
        train_op = optimizer.apply_gradients(new_grad_var)

        # 训练
        num_user =self.input_v.shape[0]
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            
            pbar=tqdm(range(self.training_epochs))
            for epoch in pbar:
                err_sum = 0
                per = list(range(num_user))
                random.shuffle(per)  
                per = [per[k:k + self.batch_size] for k in range(0, num_user, self.batch_size)] 
                num_batchs = len(per)

                for index in per:
                    _, c = sess.run([train_op, cost], feed_dict={Input_visible: self.input_v[index],
                                                    Input_visible_mask: self.input_v_mask[index],
                                                    Input_conditional: [self.input_c[n][index] for n in range(self.n_condition)],
                                                    Input_regular: self.input_r[index],
                                                    Input_geoWeight: self.weight_g
                                                   })
                    err_sum = err_sum + c
                err_sum = err_sum / num_batchs
                pbar.set_postfix({'loss': err_sum})
                
            # saver.save(sess, path_save_weight, global_step=epoch)
            ouput = sess.run(Output, feed_dict={Input_visible: self.input_v,
                                              Input_visible_mask: self.input_v_mask,
                                              Input_conditional: self.input_c,
                                              Input_regular: self.input_r,
                                              Input_geoWeight: self.weight_g})
        return ouput
            
            
        