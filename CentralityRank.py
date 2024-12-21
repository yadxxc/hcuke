import numpy as np


# 类定义（【自定义】）
class DirectedCentralityRank(object):
    def __init__(self, 
                ds_name,
                doc_id,
                doc_feats,
                doc_type,
                poolings,
                sim_methods,
                len_type = '0',
                posi_type = '1',
                *hps
                ):
        self.ds_name = ds_name
        self.doc_id = doc_id
        self.doc_feats = doc_feats # 关键属性,单文档的特征，值为字典
        self.doc_type = doc_type     # 'cls' or 'pool' or 'title'
        self.can_pool = poolings['can_pooling']
        self.sent_pool = poolings['sent_pooling']
        self.doc_pool = poolings['doc_pooling']
        self.sim_c_c = sim_methods['c-c']      # 'euc','dot','cos','man'
        self.sim_d_c = sim_methods['d-c']
        self.sim_d_s = sim_methods['d-s']
        self.sim_s_c = sim_methods['s-c']
        self.sim_s_s = sim_methods['s-s']
        self.len_type = len_type   # 0,1,2 : 忽略，原长，开方
        self.posi_type = posi_type
        self.Hps = hps # 超参数
        
        # 填充数据  
        
        # # 默认D-S, D-C
        self.d_s_dist = self.doc_feats['d_s_dists']["{}_{}_{}_{}".format(self.doc_type, self.doc_pool, self.sent_pool, self.sim_d_s)]
        self.d_c_dist = self.doc_feats['d_c_dists']["{}_{}_{}_{}".format(self.doc_type, self.doc_pool, self.can_pool, self.sim_d_c)]
        
        # 2d
        self.sents_cans_dist = self.doc_feats['can_sent_score']["{}_{}_{}".format(self.sent_pool,self.can_pool,self.sim_s_c)]
        self.cans_self_dist_2d = self.doc_feats['cans_self_score_2d']["{}_{}".format(self.can_pool,self.sim_c_c)]
        
        # ---------------------------------------------------------------------------------------
        # 候选:8元组 候选c, (emb_max, emb_mean), c序号(相对位置), token位置, 句位置, 长度, 其它同c词位置lst, 其它同c句位置lst
        self.cans = self.doc_feats['cans']
              
        # 位置权(w,s,c), 1d数组
        self.pos_weight_word = self.doc_feats['word_pos_w']
        self.pos_weight_can = self.doc_feats['can_pos_w']
        self.pos_weight_sent = self.doc_feats['sent_pos_w']
        

    # 归一化一维数组
    def normalization(self, array_1d, epsilon=1e-10):
        max_v = np.max(array_1d)
        min_v = np.min(array_1d)
        # 防止最大值和最小值相同导致除数为0的情况
        if max_v - min_v == 0:
            normalized_array = array_1d + epsilon
        else:
            normalized_array = (array_1d - min_v) / (max_v - min_v)
        
        return normalized_array
    
    # local relevance
    def local_r(self, array_2d, lamb):
        
        threshold = np.mean(array_2d) * lamb

        array_2d[array_2d <= threshold] = 0
        return np.sum(array_2d,axis= 1)
      
    
    # 最优参数配置：False,True,True,True,'d-s-c','c-c'
    def HCUKE(self):

        # 参数设置 !!!
        g_norm = False # 固定
        l_norm = True  # 固定

        is_sent_w = True  # 句位置权
        is_all_sent = True # 上下文 True:所有句 or False:所在句
        g_mode = 'd-s-c'  # 'd-s-c', 'd-c' , 's-c' ,'none'
        l_mode = 'c-c'   # 'c-c', 'c'
       

        # fiding table
        tab_d_c_dist = self.d_c_dist
        tab_d_s_dist = self.d_s_dist
        tab_s_c_dist = self.sents_cans_dist
                
        if g_norm:
            tab_d_c_dist = self.normalization(tab_d_c_dist)
            tab_d_s_dist = self.normalization(tab_d_s_dist)
            tab_s_c_dist = self.normalization(tab_s_c_dist)

        tab_c_c_2d = self.cans_self_dist_2d # 完整
        
        if self.ds_name =='DUC2001':
            lamb = 0.9
        elif self.ds_name =='Inspec':
            lamb = 1.3
        else:
            lamb = 0.8
        

        l_deep =  self.local_r(tab_c_c_2d, lamb)
        if l_norm:
            l_deep = self.normalization(l_deep)

        # S位置权
        tab_sen_w = self.pos_weight_sent

        # 计算
        cans_score = []
        # 返回字典
        
        
        # cp, (emb_max,emb_min), c_posi, c_posi_a, c_s_posi, len, c_posi_lst, c_s_posi_lst
        for idx, (_, _, c_id, c_pos, s_pos, _, c_pos_lst, s_pos_lst) in enumerate(self.cans):
            
            g_relevance = 0
            # s_pos_lst = set(s_pos_lst) # 过滤重复，效果总体略微下降
            if is_sent_w: # 考虑句子位置权
                if g_mode == 'd-s-c':
                    if is_all_sent:  # 所有句
                        for sid in s_pos_lst:
                            g_relevance += (tab_s_c_dist[idx][sid] * tab_d_s_dist[sid] * tab_sen_w[sid])
                    else: # 当前句
                        g_relevance = tab_s_c_dist[idx][s_pos] * tab_d_s_dist[s_pos] * tab_sen_w[s_pos]

                elif g_mode == 'd-c':
                    if is_all_sent:  # 所有句
                        s_w = 0
                        for sid in s_pos_lst:
                            s_w += tab_sen_w[sid]
                        g_relevance = tab_d_c_dist[idx] * s_w
                    else:  # 当前句
                        g_relevance = tab_d_c_dist[idx] * tab_sen_w[s_pos]

                elif g_mode == 's-c':
                    if is_all_sent:  # 所有句
                        for sid in s_pos_lst:
                            g_relevance += (tab_s_c_dist[idx][sid] * tab_sen_w[sid])
                    else: # 当前句
                        g_relevance = tab_s_c_dist[idx][s_pos] * tab_sen_w[s_pos]
                else: # 不考虑全局
                    if is_all_sent:  # 所有句
                        for sid in s_pos_lst:
                            g_relevance += tab_sen_w[sid]
                    else:  # 当前句
                        g_relevance = tab_sen_w[s_pos]

            else: # 无句子位置
                if g_mode == 'd-s-c':
                    if is_all_sent:  # 所有句
                        for sid in s_pos_lst:
                            g_relevance += (tab_s_c_dist[idx][sid] * tab_d_s_dist[sid])
                    else:  # 当前句
                        g_relevance = tab_s_c_dist[idx][s_pos] * tab_d_s_dist[s_pos]

                elif g_mode == 'd-c':
                    g_relevance = tab_d_c_dist[idx]

                elif g_mode == 's-c':
                    if is_all_sent:  # 所有句
                        for sid in s_pos_lst:
                            g_relevance += (tab_s_c_dist[idx][sid])
                    else:  # 当前句
                        g_relevance = tab_s_c_dist[idx][s_pos]
                else: # 不考虑全局
                    g_relevance = 1
            
            relevance = g_relevance
            
            if l_mode == 'c-c':
                relevance *= l_deep[idx]
            cans_score.append(relevance)

        return cans_score
     
     
    # 主方法【自定义：框架不变，仅修改computing_score()方法】
    def rank(self):

        # 1.获取分数【核心】
        cans_score = self.HCUKE() 

        # 候选c, (emb_max, emb_mean), c序号(相对位置), token位置, 句位置, 长度, 其它同c词位置lst, 其它同c句位置lst
        final_score = [] # 2维列表，([候选,索引号,分数])
        len_type = self.len_type
        posi_type = self.posi_type
        for phrase, can_score in zip(self.cans, cans_score):
            if len_type == '1':
                c_len = phrase[5] # c长度
            elif len_type == '2':
                c_len = np.sqrt(phrase[5])
            else:
                c_len = 1

            if posi_type == '1': 
                c_p = phrase[2] # can idx
                p_w = self.pos_weight_can[c_p]
            elif posi_type == '2':
                w_p = phrase[3] # can start postion
                p_w = self.pos_weight_word[w_p]
            else:
                p_w = 1
            
            final_score.append((phrase[0],  c_len * p_w * can_score))
            
        
        # 2. 逆序，返回预测【代码不变】
        final_score.sort(key = lambda x: x[1], reverse = True)
        predicted_candidation = [x[0].strip() for x in final_score]
        return predicted_candidation