import os
import pickle
import json

import nltk
import numpy as np
from tqdm import tqdm

import CentralityRank as crank
import config



def res_details(out_fname, f5_lst,f10_lst,f15_lst):
    cnt = len(f5_lst)
    with open(out_fname, 'w', encoding='utf-8') as file:
        for i in range(cnt):
            item = str.format('{}, {}, {}', round(f5_lst[i],2), round(f10_lst[i],2), round(f15_lst[i],2))
            file.write('%s\n' % item) 
    
def dedup(kp_list):
    dedupset = set()
    kp_list_dedup = []
    for kp in kp_list:
        if kp in dedupset:
            continue       
        kp_list_dedup.append(kp)
        dedupset.add(kp)
    return kp_list_dedup

# 子方法：evaluate()调用
def get_score_full(candidates, references, maxDepth = 15):
    precision = []
    recall = []
    reference = dedup(references) # 去重
    candidates = dedup(candidates)
    
    # reference = set(references) # 集合自带简单去重，更一般的去重可先最小化
    ref_len = len(reference) 
    pre_len = len(candidates) 
    true_positive = 0
    for i in range(maxDepth):
        if pre_len == 0 or ref_len == 0:   # 预测为0,或真实值为0(存在没有PK或AK的样本)
            precision.append(0)
            recall.append(0)
            continue
        if pre_len > i:
            kp_pred = candidates[i]     
            if kp_pred in reference:
                true_positive += 1
            precision.append(true_positive/float(i + 1))
            recall.append(true_positive/float(ref_len))
        else:
            precision.append(true_positive/float(pre_len))
            recall.append(true_positive/float(ref_len))
    return precision, recall

# 主方法extract_summary()调用
def evaluate(candidates, references):
    precision_scores, recall_scores, f1_scores = {5:[], 10:[], 15:[]}, {5:[], 10:[], 15:[]}, {5:[], 10:[], 15:[]} # 3个字典
    for candidate, reference in zip(candidates, references):
        p, r = get_score_full(candidate, reference) 
        # res_f1 = [] # 测试使用
        for i in [5,10,15]:
            precision = p[i-1]
            recall = r[i-1]
            if precision + recall > 0:
                f1_scores[i].append((2 * (precision * recall)) / (precision + recall))
            else:
                f1_scores[i].append(0)
            precision_scores[i].append(precision)
            recall_scores[i].append(recall)

    # print("########################\nMetrics")
    
    
    # 构造导出数据
    # 场景1：仅F1
    
    f5 = np.around(np.mean(f1_scores.get(5))*100,2) # 平均
    f10 = np.around(np.mean(f1_scores.get(10))*100,2)
    f15 = np.around(np.mean(f1_scores.get(15))*100,2)
    
    res_list= [f5,f10,f15]
    return res_list
    
    # # 场景2：找超参,输出F1,P,R
    # f_list=[]
    # p_list=[]
    # r_list=[]
    # res_list = []
    # for i in precision_scores: # i为键:5,10,15
    #     print("F1@{}:{}".format(i,np.around(np.mean(f1_scores[i])*100,2)),"P@{}:{}".format(i,np.around(np.mean(precision_scores[i])*100,2)),"R@{}:{}".format(i,np.around(np.mean(recall_scores[i])*100,2)))
    #     avg_f = np.around(np.mean(f1_scores[i])*100,2)
    #     f_list.append(avg_f)
    #     avg_p = np.around(np.mean(precision_scores[i])*100,2)
    #     p_list.append(avg_p)   
    #     avg_r = np.around(np.mean(recall_scores[i])*100,2)    
    #     r_list.append(avg_r) 
        
    # for res in zip(f_list,p_list,r_list):
    #     res_list.extend(list(res))  
    # return res_list # [f5,p5,r5,f10,p10,p15,r5,r10,r15] 
 
# --------------------------------------------------------------------------------------------------------------
# 主方法【核心框架，基本不变】
porter = nltk.PorterStemmer()      

# hps为元组
def KP_Extract(ds_name, data_base_dir, docs_feats, doc_type, poolings, sim_methods, len_type, posi_type, *hps):
    data= "{}/{}/{}_test.json".format(config.DATA_SET_DIR, ds_name,ds_name)
    with open(data, 'r') as f:
        data_lines = f.readlines()
    groundtruth_kps = []
    # step 1: 预测
    predicted_candidations = [] # 2d
    for doc_id, doc_feats in enumerate(tqdm(docs_feats)):
        #------------------------------预测-----------------------------
        ranker = crank.DirectedCentralityRank(ds_name, doc_id, doc_feats, doc_type, poolings, sim_methods, len_type, posi_type, hps)
        pred_kps = ranker.rank()

        # 去重
        pred_kps = dedup(pred_kps)
        predicted_candidations.append(pred_kps)

        groundtruth_kps.append(json.loads(data_lines[doc_id])['PKs'])

        
    # step 2: 创建新目录，保存预测
    pred_dir = os.path.join(data_base_dir, "pred")
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    predicted_fname = "pooling-{}_doc-{}_predicted.txt".format(poolings['can_pooling'], doc_type)

    with open(os.path.join(pred_dir, predicted_fname), 'w', encoding='utf-8') as f:
        for s in predicted_candidations:
            f.writelines("; ".join(s[:20]) + "\n")

    # step 3: 评价
    predicted_candidation_stem = []
    for phrases in predicted_candidations:
        cand = []
        for phrase in phrases:        
            phrase = " ".join([porter.stem(x) for x in phrase.strip().split(' ')])
            cand.append(phrase)
        predicted_candidation_stem.append(cand)
    
    eval_score = evaluate(predicted_candidation_stem, groundtruth_kps) # groundtruth_kps 数据集中已词干化
    return eval_score


if __name__ == "__main__":
    # # # 配置
    model= config.EMB_MODEL
    ds_names=config.DATA_SETS
        
    # # 1-三数据集一次性全部运行

    # 'cos','dot','man','euc'
    sim_methods = {}  
    sim_methods['d-s'] = 'cos'
    sim_methods['s-c'] = 'cos'
    sim_methods['c-c'] = 'cos'
    sim_methods['d-c'] = 'man'  # 暂无用
    sim_methods['s-s'] = 'cos'  # 暂无用
   
    doc_type = 'doc'   # 
    poolings = {}  
    poolings['doc_pooling'] = 'max'  #  + 'cls'
    poolings['sent_pooling'] = 'max' # 'max' or 'mean'
    poolings['can_pooling'] = 'max'  # 'max' or 'mean'

    posi_type = '1' # 位置'0' or '1' or '2' (0为忽略，1为C序号，2为C起始位)
    len_type = '0'  # 长度'0' or '1' or '2' (0为忽略，1为短语长，2为平方根)

    res_list = []
    for  ds_name in ds_names:
        data="{}/{}".format(config.DATA_SET_DIR,ds_name)
        # 加载数据
        try:
            with open(os.path.join(data, f"data.{ds_name}.{model}.{poolings['can_pooling']}.pkl"), 'rb') as f:
                docs_feats = pickle.load(f)
        except: 
            print("读取文档嵌入错误,请检查嵌入文件是否存在！")
        res_list.extend(KP_Extract(ds_name, data, docs_feats, doc_type, poolings, sim_methods, len_type, posi_type))
    print(res_list)
       