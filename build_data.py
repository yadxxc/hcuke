# -*- coding: UTF-8 -*-
"""
@Project ：HCUKE2023 
@File ：preparing_datas.py
@Author ：BIT.XC
@Date ：2023/8/2 8:46 
"""
import sys
import os
import pickle
from tqdm import tqdm
import numpy as np

import config
import utils

# region 相似性计算 -*-*-*-*--*-*--*-*--*-*-*-*--*-*--*-*--*-*-*-*--*-*--*-*--*-*-*-*--*-*--*-*-

# 1) 文档-短语/句子(1:n -> n):用作权重；返回：1d数组
def compute_doc_sentsorcans_similarity(sentsorcans_embs, doc_emb, method='man'):
    cnt_cans = len(sentsorcans_embs)
    res_array = np.zeros(cnt_cans)
    for idx in range(cnt_cans):
        res_array[idx] = utils.vec_vec_sim(sentsorcans_embs[idx], doc_emb, method)
    return res_array

# 2) 句子-句子 或 候选-候选：用于计算每个句子/短语顶点的初始重要性；返回：1d数组
# 公共方法(2d np.array)
def inner_score(embs, method = 'dot', is_skip_self= True):
    ndim = len(embs)
    score_array = np.zeros((ndim,ndim)) # 初始值
 
    for ridx in range(ndim):
        for cidx in range(ndim):
            if is_skip_self and ridx == cidx: # 是否跳过自身
                continue
            if method in ['cos', 'dot']:
                score_array[ridx][cidx] = utils.vec_vec_sim(embs[ridx], embs[cidx], method)
            else:
                sim = utils.vec_vec_sim(embs[ridx], embs[cidx], method)
                score_array[ridx][cidx] = sim
    return score_array

# 方案1：句子/短语(1d) 
def compute_inner_score(embs, method = 'dot', is_skip_self= True):
    inner_sim = inner_score(embs, method, is_skip_self)
    score_array = np.sum(inner_sim, axis= 1) # 按行求和
    return score_array

# 方案2：过滤1 句子/短语 除自身外，其它自身点乘 * 它们间距离 [HGUKE2023]
def compute_inner_score2(embs, method = 'dot', is_skip_self = True):
    ndim = len(embs)

    inner_sim = inner_score(embs, method, is_skip_self)
    avg_score = np.mean(inner_sim)
    
    for ridx in range(ndim):
        for cidx in range(ndim):
            if is_skip_self and ridx == cidx: # 不含自身点乘
                continue
            inner_sim[ridx][cidx] = inner_sim[ridx][cidx] - 0.9 * avg_score
    
    return np.sum(inner_sim, axis= 1) 

# 方案3：过滤2    [Joint2023]
def compute_inner_score3(embs, method = 'dot', is_skip_self = False):
    ndim = len(embs)
    inner_sim = inner_score(embs, method, is_skip_self)
    min_score = np.min(inner_sim) # 矩阵元素 最小值
    max_score = np.max(inner_sim)
    threshold = 0.2 * (max_score - min_score)  # 论文第4页，阈值theta，与论文不一致(原代码：min_score + self.beta * (max_score - min_score))
    
    for ridx in range(ndim):
        for cidx in range(ndim):
            if is_skip_self and ridx == cidx: # 不含自身点乘
                continue
            tmp_val = inner_sim[ridx][cidx] - threshold
            tmp_val = tmp_val if tmp_val>=0 else 0
            inner_sim[ridx][cidx] = tmp_val
    return np.sum(inner_sim, axis= 1) 

# 3) 句子-候选(点乘):用作权重；返回：2d数组(行对应短语,列对应句子)
def compute_sents_cans_similarity(cans_emb, sents_emb, method='man'):
    r_num = len(cans_emb) # 候选
    c_num = len(sents_emb) # 句子
    sim_mat = np.zeros((r_num, c_num))
    for ridx, can_emb in enumerate(cans_emb):
        for cidx, sent_emb in enumerate(sents_emb):
            sim_mat[ridx][cidx] = utils.vec_vec_sim(can_emb, sent_emb, method)
    
    return sim_mat

# 4)计算位置分
def position_weight(total_cnt):
    """
    return: 1d array
    """
    pos_weight = 1 / (np.array(list(range(1, total_cnt + 1))))  # 位置权重，1,1/2,...,1/n.其中,n为当前文档候选的个数
    pos_weight = np.exp(pos_weight) / np.sum(np.exp(pos_weight))  # >>>>>>>>>>>论文公式8>>>>>>>>>>>
    return pos_weight

# 5)表面相似性
def surface_sim(cans, lambda1 = 0):
    # 计算表面相似性,[并计算每个候选出现的位置(候选位置,句子位置)]
    cnt_phrase = len(cans)
    surf_sim = np.zeros((cnt_phrase,cnt_phrase))
    for idx in range(cnt_phrase):
        surf_sim[idx][idx] = 1
        for jidx in range(0,cnt_phrase):
            if idx == jidx:
                continue
            surf_sim[idx][jidx] = utils.simple_word_overlap_similarity(cans[idx],cans[jidx],True,True,lambda1)
    
    score_array = np.sum(surf_sim, axis= 1) # 按行求和
    return score_array, surf_sim


# 6)列表排序并返回序号
def sort_list(original_list, reverse):
    # 使用sorted()函数进行排序，并通过list comprehension获取排序后的序号
    sorted_list_with_indices = sorted(enumerate(original_list), key=lambda x: x[1], reverse=reverse)
    
    sorted_list = [element for ( _, element) in sorted_list_with_indices]
    sorted_idx = [index for (index, _ ) in sorted_list_with_indices]
    return sorted_list, sorted_idx
# endregion -*-*-*-*--*-*--*-*--*-*-*-*--*-*--*-*--*-*-*-*--*-*--*-*--*-*-*-*--*-*--*-*-

# 7)获取句子新位置(默认句子索引是[0,1,2..];按降序)
def rerank_sents(original_list):
    """
    original_list: 重要性分
    return: 原位置(0,1,...,i,...) ->新位置lst[i]
    """
    sorted_idx_list = sort_list(original_list,True)[1] #
    new_s_pos = []
    for i in range(len(sorted_idx_list)):
        new_s_pos.append(sorted_idx_list.index(i))
    return new_s_pos

# 归一化一维数组
def normalization(array_1d, epsilon=1e-10):
    max_v = np.max(array_1d)
    min_v = np.min(array_1d)
    # 防止最大值和最小值相同导致除数为0的情况
    if max_v - min_v == 0:
        normalized_array = array_1d + epsilon
    else:
        normalized_array = (array_1d - min_v) / (max_v - min_v)
    
    return normalized_array


if __name__=="__main__":
    ds_name = sys.argv[1]
    pooling = sys.argv[2]
    extract = sys.argv[3] # 1 效果好！
    # sim_method = sys.argv[4]
    
    # ds_name = 'SemEval2010' #'Inspec','SemEval2010','DUC2001'
    # pooling = 'max'
    # extract = '1'   # 整体(0) or 按句(1)
    # # # sim_method = 'cos' # 'euc' or 'dot' or 'cos' or 'man' ,候选间 or 句间
    #-----------------------------------------------------------------------
    
    data="{}/{}".format(config.DATA_SET_DIR,ds_name)
    model = config.EMB_MODEL
    
    pool_idx = {"max": 0, "mean":1}
    p_idx = pool_idx[pooling]
    
    extr_keys = {"0": "d_can_author", "1": "d_sent_cans"}
    extr_key = extr_keys[extract]
    
    # joint2023
    if ds_name in ['DUC2001', 'Inspec']:
        lambda1 = 0.2
    else:
        lambda1 = 1
    
    # 加载数据
    try:
        with open(os.path.join(data, f"test.alllevel.feats.{model}.pkl"), 'rb') as f:
            docs_feats = pickle.load(f)
    except:
        print("读取文档嵌入错误,请检查嵌入文件是否存在！")
    
    data_dict_lst = []  # 字典列表
    # 填充数据------------------------------------------------------------------------
    for doc_feats in tqdm(docs_feats):
        # -------------------------T-A(doc)整体-------------------------
        # 1) doc(表示)
        d_emb_cls = doc_feats['d_cls_emb']
        d_emb_bertcls = doc_feats['d_bertcls_emb']  
        
        # 2元组(max, mean)
        d_emb = doc_feats['d_emb']                 # 基于整个doc池化
        d_emb_t = doc_feats['d_emb_t']             # 基于标题or第一句池化 (HGUKE2023平均池化)
        d_emb_erank = doc_feats['d_emb_embedrank'] # 基于候选词池化 (Embedrank2018平均)
        # d_emb_embedrank_author = doc_feats['d_emb_embedrank_author'][p_idx] # 整体抽候选
        
        # 2）sent(表示)
        d_sent_embs = doc_feats['d_sent_embs']  # -2元组

        # 3）can 8元组(候选c, (emb_max, emb_mean), token位置, c序号(相对位置), 句位置, 长度, 其它同c词位置lst, 其它同c句位置lst)
        # 1 【整体抽取-key: 'd_can_author'】(缺点:难于获取短语对应的句子信息) 
        # 2 【按句抽取-key: 'd_sent_cans'】(a.便于获取短语对应的句子信息；b.候选可能略有差异！)
        d_can_author = doc_feats[extr_key][p_idx] # -2元组
        
        # d_can_author = doc_feats['d_can_author'][p_idx]
        # d_sent_cans = doc_feats['d_sent_cans'][p_idx]
        
        # 准备数据 --------------------------------------------------------------------------------------
        # 解压候选嵌入  8元组(候选,(emb_max, emb_mean), can_id, token位置, 句位置, 长度, token位置lst, 句位置lst)
        cp, can_embs, _, _, _, _, _, _= list(zip(*d_can_author))
        
        # 计算句子重要性【3种方案：D-S相似、S-S中心性、(D-S) - (D-S)中心性，暂采用第1种】    
        
        # D-C/S相似性(4种:euc,dot,cos,man)
        # D的表示：cls, pool(整文档, 最重要的句子(标题), 第1个句子, 所有候选)
        d_c_dists = {}
        d_s_dists = {}

        d_tidx_c = {} # 标题索引
        d_tidx_max_p = {}
        d_tidx_mean_p = {}
        
        d_s_sort_idx_cls = {} # 有序句idx
        d_s_sort_idx_max_pool = {}
        d_s_sort_idx_mean_pool = {}
        
        c_s_dist = {} # C-S 相似性
        s_s_dist_2d = {} # S-S 相似性
        s_s_dist = {} # S-S 相似性
        c_c_dist_2d = {} # C-C 相似性
        c_c_dist = {} # C-C 相似性

        for sim_m in ['euc', 'dot', 'cos', 'man']:

            for idx, s_pool in enumerate(['max','mean']):
                # 余弦可能会导致负值
                # D - S
                keystr = "{}_{}_{}" + f"_{sim_m}" # d_txt, d_pool, s_pool/c_pool, sim

                cls_keystr = keystr.format('doc', 'cls', s_pool)
                max_keystr = keystr.format('doc', 'max', s_pool)
                mean_keystr = keystr.format('doc', 'mean', s_pool)
                # D - S
                d_s_dists[cls_keystr] = compute_doc_sentsorcans_similarity(d_sent_embs[idx], d_emb_cls, sim_m)
                d_s_dists[max_keystr] = compute_doc_sentsorcans_similarity(d_sent_embs[idx], d_emb[0], sim_m)
                d_s_dists[mean_keystr] = compute_doc_sentsorcans_similarity(d_sent_embs[idx], d_emb[1], sim_m)

                # (3套方案)找最重要句子:D-S相似性(2)和S中心性(1)
                d_s_sort_idx_cls[cls_keystr] = rerank_sents(d_s_dists[cls_keystr])
                d_s_sort_idx_max_pool[max_keystr] = rerank_sents(d_s_dists[max_keystr])
                d_s_sort_idx_mean_pool[mean_keystr] = rerank_sents(d_s_dists[mean_keystr])

                # a. S - D 最相似（2个）
                max_s_idx_c = d_s_sort_idx_cls[cls_keystr][0] # 第1个最重要
                max_s_idx_max_p = d_s_sort_idx_max_pool[max_keystr][0]
                max_s_idx_mean_p = d_s_sort_idx_mean_pool[mean_keystr][0]

                d_tidx_c[cls_keystr] = max_s_idx_c
                d_tidx_max_p[max_keystr] = max_s_idx_max_p
                d_tidx_mean_p[mean_keystr] = max_s_idx_mean_p

                title_c = d_sent_embs[idx][max_s_idx_c]
                title_max = d_sent_embs[idx][max_s_idx_max_p]
                title_mean = d_sent_embs[idx][max_s_idx_mean_p]
                # b. S-S中心性(1个)
                # c. (D-S)-(D-S)中心性(1个)

                cls_keystr = keystr.format('title', 'cls', s_pool)
                max_keystr = keystr.format('title', 'max', s_pool)
                mean_keystr = keystr.format('title', 'mean', s_pool)
                # D - S
                d_s_dists[cls_keystr] = compute_doc_sentsorcans_similarity(d_sent_embs[idx], title_c, sim_m)
                d_s_dists[max_keystr] = compute_doc_sentsorcans_similarity(d_sent_embs[idx], title_max, sim_m)
                d_s_dists[mean_keystr] = compute_doc_sentsorcans_similarity(d_sent_embs[idx], title_mean, sim_m)

                max_keystr = keystr.format('fsent', 'max', s_pool)
                mean_keystr = keystr.format('fsent', 'mean', s_pool)
                # D - S
                d_s_dists[max_keystr] = compute_doc_sentsorcans_similarity(d_sent_embs[idx], d_emb_t[0], sim_m)
                d_s_dists[mean_keystr] = compute_doc_sentsorcans_similarity(d_sent_embs[idx], d_emb_t[1], sim_m)

                max_keystr = keystr.format('erank', 'max', s_pool)
                mean_keystr = keystr.format('erank', 'mean', s_pool)
                # D - S
                d_s_dists[max_keystr] = compute_doc_sentsorcans_similarity(d_sent_embs[idx], d_emb_erank[0], sim_m)
                d_s_dists[mean_keystr] = compute_doc_sentsorcans_similarity(d_sent_embs[idx], d_emb_erank[1], sim_m)

                # D - C  # d_txt, d_pool, c_pool(与pooling一致), sim
                cls_keystr = keystr.format('doc', 'cls', pooling)
                max_keystr = keystr.format('doc', 'max', pooling)
                mean_keystr = keystr.format('doc', 'mean', pooling)
                d_c_dists[cls_keystr] = compute_doc_sentsorcans_similarity(can_embs, d_emb_cls, sim_m)
                d_c_dists[max_keystr] = compute_doc_sentsorcans_similarity(can_embs, d_emb[0], sim_m)
                d_c_dists[mean_keystr] = compute_doc_sentsorcans_similarity(can_embs, d_emb[1], sim_m)

                # D - C
                cls_keystr = keystr.format('title', 'cls', pooling)
                max_keystr = keystr.format('title', 'max', pooling)
                mean_keystr = keystr.format('title', 'mean', pooling)
                d_c_dists[cls_keystr] = compute_doc_sentsorcans_similarity(can_embs, title_c, sim_m)
                d_c_dists[max_keystr] = compute_doc_sentsorcans_similarity(can_embs, title_max, sim_m)
                d_c_dists[mean_keystr] = compute_doc_sentsorcans_similarity(can_embs, title_mean, sim_m)

                # D - C
                max_keystr = keystr.format('fsent', 'max', pooling)
                mean_keystr = keystr.format('fsent', 'mean', pooling)
                d_c_dists[max_keystr] = compute_doc_sentsorcans_similarity(can_embs, d_emb_t[0], sim_m)
                d_c_dists[mean_keystr] = compute_doc_sentsorcans_similarity(can_embs, d_emb_t[1], sim_m)


                # D - C
                max_keystr = keystr.format('erank', 'max', pooling)
                mean_keystr = keystr.format('erank', 'mean', pooling)
                d_c_dists[max_keystr] = compute_doc_sentsorcans_similarity(can_embs, d_emb_erank[0], sim_m)
                d_c_dists[mean_keystr] = compute_doc_sentsorcans_similarity(can_embs, d_emb_erank[1], sim_m)


                # C-S / S-S / C-C
                # 2d
                keystr2 = "{}_{}"  # d_txt, d_pool, s_pool/c_pool, sim
                s_s_dist_2d[keystr2.format(s_pool, sim_m)] = inner_score(d_sent_embs[idx], sim_m, is_skip_self= False) # 不跳过;完整
                c_c_dist_2d[keystr2.format(pooling, sim_m)] = inner_score(can_embs, sim_m, is_skip_self= False)
                # 1d [最简单标准中心性(不含自身) - 全连接]
                s_s_dist[keystr2.format(s_pool, sim_m)] = compute_inner_score(d_sent_embs[idx], sim_m, is_skip_self=True)  # 跳过
                c_c_dist[keystr2.format(pooling, sim_m)] = compute_inner_score(can_embs, sim_m, is_skip_self=True)

                c_s_dist["{}_{}_{}".format(s_pool,pooling,sim_m)] = compute_sents_cans_similarity(can_embs, d_sent_embs[idx], sim_m)

            
        datas_dict = {}
        # 构造数据字典
        cnt_can = len(d_can_author)
        cnt_sent = len(d_sent_embs[0]) # d_sent_embs为2元组
        
        datas_dict['cnt_can'] = cnt_can
        datas_dict['cnt_sent'] = cnt_sent
        # D-C
        datas_dict['d_c_dists'] = d_c_dists
        # D-S
        datas_dict['d_s_dists'] = d_s_dists

        
        datas_dict['d_tidx_c'] = d_tidx_c  # 受相似方法影响明显
        datas_dict['d_tidx_max'] = d_tidx_max_p  # 比较稳定
        datas_dict['d_tidx_mean'] = d_tidx_mean_p
        
        datas_dict['d_s_sort_idx_cls'] = d_s_sort_idx_cls
        datas_dict['d_s_sort_idx_max_pool'] = d_s_sort_idx_max_pool
        datas_dict['d_s_sort_idx_mean_pool'] = d_s_sort_idx_mean_pool
      
        # C-S / S-S / C-C / Cs- Cs
        datas_dict['can_sent_score'] = c_s_dist
        datas_dict['sents_self_score_2d'] = s_s_dist_2d
        datas_dict['sents_self_score'] = s_s_dist  # 1d
        datas_dict['cans_self_score_2d'] = c_c_dist_2d
        datas_dict['cans_self_score'] = c_c_dist   # 1d
        
        s_sim_1d, s_sim_2d = surface_sim(cp) # 0
        datas_dict['surface_sim'] = s_sim_1d
        datas_dict['surface_sim_2d'] = s_sim_2d
        s_sim25_1d, s_sim25_2d = surface_sim(cp, 0.25) # 阈值0.25
        datas_dict['surface_sim25'] = s_sim25_1d
        datas_dict['surface_sim25_2d'] = s_sim25_2d
        datas_dict['can_pos_w'] = position_weight(cnt_can) # 选短语的相对位置计算
        datas_dict['word_pos_w'] = position_weight(doc_feats['d_tokens_cnt']) # PositionRank
        datas_dict['sent_pos_w'] = position_weight(cnt_sent)
        datas_dict['cans'] = d_can_author
        # datas_dict['cans2'] = d_sent_cans
        datas_dict['keyphrases'] = doc_feats['d_keyphrases']
        
        data_dict_lst.append(datas_dict)


    # 保存数据文件
    docs_datas_kpl_path = os.path.join(data, f"data.{ds_name}.{model}.{pooling}.pkl")
    with open(docs_datas_kpl_path, 'wb') as f:
        pickle.dump(data_dict_lst, f)