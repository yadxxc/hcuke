import os
import json
import pickle
import sys 

import torch

from tqdm import tqdm
from transformers import BertTokenizer, BertModel

import config

def print_run_time(func):
    import time
    def wrapper(*args, **kw):
        local_time = time.time()
        res = func(*args, **kw)
        print("Current function [%s] run time is %.8f (s)" % (func.__name__, time.time() - local_time))
        return res
    return wrapper

@print_run_time
def read_json_by_line(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines  = f.readlines()
    return [json.loads(line.strip()) for line in lines]

# 方法1：核心方法（1d词列表）
def doc_embedding(tokenizer, model, tokens):
    token_subwords_lens = [] # token分为子词的个数
    input_tokens = ['[CLS]'] # 构造bert格式的输入(含CLS和SEP特殊token)
    
    for token in tokens:
        token_subwords = tokenizer.tokenize(token) # bert子词
        
        if len(input_tokens) + len(token_subwords) >= 511:
            break
        else:
            input_tokens.extend(token_subwords)
            token_subwords_lens.append(len(token_subwords))
    input_tokens += ["[SEP]"]
    
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens) # 1d列表
    input_ids = torch.LongTensor([input_ids]) # 2d Tensor,1 x n
    
    # subwords_embs：(1,n,768),tensor, 其中，n为含特殊符号的token个数【重要】
    # cls_embs:(1,768),tensor,'[CLS]'嵌入过线性层后的结果,可应用于分类等下游任务【暂时没用】
    # layer_embs:长度13的元组,每个元素为(1,n,768)的tensor,第0层为输入嵌入，后面12个为各层输出
    # 第1维最大可以是2(句对)【暂时没用】
    subwords_embs, cls_embs, layer_embs = model(input_ids, output_hidden_states=True) 
    
    bert_cls_emb  = cls_embs.squeeze().detach().numpy() # 768 一维数组
    
    subwords_embs = subwords_embs.squeeze().detach().numpy() # n x 768 二维数组
    cls_token_emb = subwords_embs[0] # 768 一维数组,原始cls嵌入
    
    # 计算各token嵌入
    tokens_emb = [] # token嵌入为各子词嵌入的平均
    idx = 1 # 跳过cls的嵌入
    for token_subword_len in token_subwords_lens:
        if token_subword_len == 1:
            tokens_emb.append(subwords_embs[idx])
            idx += 1
        else:
            tokens_emb.append(sum(subwords_embs[idx:idx+token_subword_len]) / token_subword_len)  # 子词平均
            # tokens_emb.append(np.max(np.array(o1[i: i+j]), axis=0))
            idx += token_subword_len
            
    assert len(tokens_emb) == len(token_subwords_lens)
    # return type: list,numpy.ndarray,numpy.ndarray
    return tokens_emb, bert_cls_emb, cls_token_emb

# 辅助方法，列表降维
def flat_list(hd_list):
    return [x for ld_list in hd_list for x in ld_list]


# 方法2：文档嵌入-主方法
def documents_embeddings(json_list, tokenizer, model, tg_path):
    """
    is_whole: bool 标题与摘要是否整体计算嵌入
    """
    docs_embeddings = []
    for doc in tqdm(json_list, desc="Encoding docs..."): 
        dic = {}
        token_2dlst = doc['tokens']  # 2d list
        tokens = flat_list(token_2dlst)  # (T+A or Doc)
        tokens_pos = flat_list(doc['tokens_pos'])

        dic['d_id'] = doc['document_id']
        dic['d_tokens'] = tokens              #  1d list
        dic['d_tokens_pos'] = tokens_pos      #  1d list
        dic['d_keyphrases'] = doc['keyphrases']
        
        sents_len = [len(x) for x in token_2dlst] # 每个句子token数, 用于获取句子信息
        dic['d_sents_len'] = sents_len
        
        # T-A 整体
        tokens_embs, bertcls_emb, cls_token_emb = doc_embedding(tokenizer, model, tokens) # 调用单文档嵌入方法
        dic['d_cls_emb'] = cls_token_emb
        dic['d_bertcls_emb'] = bertcls_emb
        dic['d_tokens_embs'] = tokens_embs
        
        docs_embeddings.append(dic)

    # 保存嵌入
    with open(tg_path, 'wb') as f:
        pickle.dump(docs_embeddings, f)
    
    return docs_embeddings


if __name__=="__main__":
    # ds_name = 'Inspec' # DUC2001、Inspec、SemEval2010、kp20k、Krapivin、NUS
    ds_name = sys.argv[1]
    
    file_path="{}/{}".format(config.DATA_SET_DIR,ds_name)
    file_name="test.json"  # 作者提供(已预处理)
    
    # model_name="/home/xc/bert-base-uncased"
    model_name=config.EMB_MODEL

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    docs_embs_kpl_path = os.path.join(file_path, f"test.doclevel.embeddings.{model_name}.pkl")
    if not os.path.exists(docs_embs_kpl_path):
        json_data_list = read_json_by_line(os.path.join(file_path, file_name))
        docs_embeddings = documents_embeddings(json_data_list, tokenizer, model, docs_embs_kpl_path) # 返回并保存至文件
    # else:
    #     with open(docs_embs_kpl_path, 'rb') as f:
    #         docs_embeddings = pickle.load(f)