import os
import pickle
import sys 
import nltk
from nltk.corpus import stopwords
import numpy as np
from tqdm import tqdm
import config


stopword_dict = set(stopwords.words('english'))
porter = nltk.PorterStemmer()    

# 【通用方法】：Embedding, 依起止位置(2元组)查找并计算
def compute_embs(doc_embs, start_end_tuple, pooling = 'max'):
    """
    嵌入池化
    """
    start_idx, end_idx = start_end_tuple
    # 单个词
    if start_idx == end_idx: # 理论不存在，至少end大1
        return doc_embs[start_idx]
    
    # 边界最大化方法（高于平均低于最大）
    # if end_idx - start_idx == 1:
    #     emb_np = np.max(np.array(doc_embs[start_idx:end_idx]), axis=0)
    # else:
    #     emb_np = np.max((doc_embs[start_idx],doc_embs[end_idx-1]), axis=0)
    
    embs = np.array(doc_embs[start_idx:end_idx])
    if pooling == 'mean':
        emb_np = np.mean(embs, axis=0) # 效果最差
    else:
        emb_np = np.max(embs, axis=0) # 作者代码
    return emb_np
    

# 子方法2.1：【抽取候选】（no_subset参数未使用）-作者方法（一次性抽取，不分句offset = 0）
# 另分句用法：offset非零，即句子标注 + 句子偏移量(计算短语的起止位置)
def extract_candidates(tokens_tagged, offset = 0, no_subset = False):
    """
    Based on POS return a list of candidate phrases
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of tuple: [(string, (start_index, end_index))]
    """
    
    GRAMMAR1 = """NP:{<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""
    np_parser = nltk.RegexpParser(GRAMMAR1)  # Noun phrase parser
    
    keyphrase_candidate = []
    np_parse_tree = np_parser.parse(tokens_tagged)

    for node in np_parse_tree:
        if (isinstance(node, nltk.tree.Tree) and node._label == "NP"):
            np = ' '.join(word for word, tag in node.leaves())
            np_len = len(node.leaves())
            start_end_idx = (offset, offset + np_len)
            keyphrase_candidate.append((np, start_end_idx))
            offset += np_len
        else:
            offset += 1

    return keyphrase_candidate

# 子方法2：Cans及位置==》Cans嵌入
def cans_embeddings(doc_embeddings, can_phrases):
    """
    can_phrases: 二元组列表,('can',(start,end))
    """
    can_embs_max = []  # 候选短语(嵌入)
    can_embs_mean = [] 
    for _, start_end_tuple in can_phrases:
        start_idx,end_idx = start_end_tuple
        if end_idx <= start_idx or end_idx > len(doc_embeddings): # 超过嵌入长度
            break
        can_embs_max.append(compute_embs(doc_embeddings,start_end_tuple,'max'))
        can_embs_mean.append(compute_embs(doc_embeddings,start_end_tuple,'mean'))
    return can_embs_max, can_embs_mean

# 子方法2：Cans及位置==》Cans嵌入
def doc_embeddings(doc_embeddings, can_phrases):
    """
    can_phrases: 二元组列表,('can',(start,end))
    """
    can_embs = []      # 所有候选短语的词向量
    for _, start_end_tuple in can_phrases:
        start_idx,end_idx = start_end_tuple
        if end_idx <= start_idx or end_idx > len(doc_embeddings): # 超过嵌入长度
            break
        can_embs.extend(doc_embeddings[start_idx:end_idx]) 
    doc_emb_max = np.max(np.array(can_embs),axis=0)
    doc_emb_mean = np.mean(np.array(can_embs),axis=0)
    return doc_emb_max, doc_emb_mean


# 辅助方法：获取句子的起止位置(含截断处理)
def sent_position(sents_len, max_len):
    doc_sents_lens = [] # 句子起止位置
    offset = 0
    for slen in sents_len:
        if offset + slen >= max_len:
            doc_sents_lens.append((offset, max_len)) # 后面的截断
            break
        else:
            doc_sents_lens.append((offset, offset+slen))
        offset += slen
    return doc_sents_lens

# 抽取1：一次抽 - 从整个文档的嵌入、tokens和POS中抽取
def get_cans_embeddings(doc_embeddings, tokens_tags):
    """
    return：句子嵌入、候选所在句子、候选及其嵌入
    """
    cans = []         # 候选 
    cans_embs_max =[]     # 候选嵌入
    cans_embs_mean =[] 
    # tmp = [] 
    cans = extract_candidates(tokens_tags)
    cans_embs_max, cans_embs_mean = cans_embeddings(doc_embeddings, cans)

    return cans_embs_max, cans_embs_mean

# 抽取2：按句抽取
def get_cans_embeddings(doc_embeddings, tokens_tags, sents_len):
    """
    return：句子嵌入、候选所在句子、候选及其嵌入
    """
    # 每个句子的起止位置（由2维tokens列表获取）
    max_len = len(doc_embeddings) # doc 长度(<=实际长度)
    sents_pos_idx = sent_position(sents_len, max_len)
    
    cans_s = []         # 候选 
    cans_embs_s_max =[]     # 候选嵌入
    cans_embs_s_mean =[] 
    # tmp = [] 
    for idx, sent_offset in enumerate(sents_pos_idx):
        start, end= sent_offset
        tmp_cans = extract_candidates(tokens_tags[start:end], start)
        temp_cans_emb_max, temp_cans_emb_mean = cans_embeddings(doc_embeddings, tmp_cans)
        cans_s.extend(tmp_cans)
        cans_embs_s_max.extend(temp_cans_emb_max)
        cans_embs_s_mean.extend(temp_cans_emb_mean)
        # cans_sentidx.extend([idx]*len(tmp_cans)) # 候选对应的句子索引
    
    return cans_s, cans_embs_s_max, cans_embs_s_mean

# 获取sents嵌入
def get_sents_embeddings(doc_embeddings, sents_len):
    d_sents_embs_max = []
    d_sents_embs_mean = []
    max_tokens = len(doc_embeddings)
    sents_pos_idx = sent_position(sents_len, max_tokens)
    # 每个句子的起止位置（由句子长度获取）
    token_sent_idx = []  # token_idx -> sent_idx
    for sidx, (begin, end) in enumerate(sents_pos_idx):
        s_emb = doc_embeddings[begin : end]
        d_sents_embs_max.append(np.max(np.array(s_emb),axis=0))
        d_sents_embs_mean.append(np.mean(np.array(s_emb),axis=0))
        token_sent_idx.extend([sidx] * (end - begin))
    return d_sents_embs_max, d_sents_embs_mean, token_sent_idx

# 子方法1：构造POS二元组（含处理停用词）
def prepare_word_tags(tokens_1dlist, pos_tags_1dlist):
    tokens_tags = list(zip(tokens_1dlist, pos_tags_1dlist))
    
    for i, token in enumerate(tokens_1dlist):
        if token.lower() in stopword_dict:
            tokens_tags[i] = (token, "IN")
    return tokens_tags

# 返回 7元组列表(加入can所在句子列表，并根据嵌入个数去除多余can)
def extend_cans(doc_cans, can_embs, token2sent_idx):
    """
    doc_cans: 2元组列表
    can_embs: np数组列表
    token2sent_idx: token_idx -> sent_idx
    return: 8(cp, (emb_max,emb_mean), c_posi, c_posi_a, c_s_posi, len, c_posi_lst, c_s_posi_lst
    """
    doc_cans_new = [] # **候选的列表次序会改变**
    finished_lst = [] # 已处理的can id
    doc_cans = doc_cans[:len(can_embs)]
    
    for c_idx, (can, (start, _ )) in enumerate(doc_cans):
        if c_idx in finished_lst:
            continue
        
        # 找相同can(用列表保持数据次序一致)
        c_idx_lst = []    # 候选索引
        c_idx_lst.append(c_idx)
        s_idx_lst = []    # 句子索引
        s_idx = token2sent_idx[start]
        s_idx_lst.append(s_idx) # 当前can
        c_posi_lst = []   # 起始位置
        c_posi_lst.append(start) 
        for c_idx2 in range(c_idx + 1, len(doc_cans)):
            if c_idx2 in finished_lst:
                continue
            
            can2, (start2, _ ) = doc_cans[c_idx2]
            if can2.strip().lower() == can.strip().lower():    # 字面相同, 也可词干,语义相同
                s_idx_lst.append(token2sent_idx[start2])
                c_idx_lst.append(c_idx2)
                c_posi_lst.append(start2)
        
        # cp_stem = " ".join([porter.stem(x) for x in can.strip().split(' ')])
        for j, cid in enumerate(c_idx_lst):
            cp = doc_cans[cid][0].strip()
            cp_len = len(cp.split())
            # cp, (emb_max,emb_min), c_posi, c_posi_a, c_s_posi, len, c_posi_lst, c_s_posi_lst
            doc_cans_new.append((cp, can_embs[cid], cid, c_posi_lst[j], s_idx_lst[j], cp_len, c_posi_lst, s_idx_lst)) # 7元组
        finished_lst.extend(c_idx_lst)
    return doc_cans_new

# 主方法： 文档、句子和候选嵌入
def get_sents_cans_embs(docs_embeddings):
    """
    在token级表示的基础上，提取句子表示和短语表示
    """
    docs_feats = []
    # cans_lst = [] # Test
    diff_idx = [] # 整体抽与分句抽结果不一致的索引号
    for d_id, doc_embs in tqdm(enumerate(docs_embeddings), total=len(docs_embeddings)):    
        # doc
        doc_tokens_embs = doc_embs['d_tokens_embs'] # 1d(元素为向量)
        
        dic = {}
        dic['d_tokens_cnt'] = len(doc_tokens_embs) # 用于生成word的位置权列表
        dic['d_keyphrases'] = doc_embs['d_keyphrases']
        
        # 1) doc 嵌入
        dic['d_cls_emb'] = doc_embs['d_cls_emb']
        dic['d_bertcls_emb'] = doc_embs['d_bertcls_emb'] # cls嵌入经过线性变换的,可应用于分类等
        
        # 基于整个文档
        d_emb_max = np.max(np.array(doc_tokens_embs),axis=0) # 按列
        d_emb_mean = np.mean(np.array(doc_tokens_embs),axis=0)
        dic['d_emb'] = (d_emb_max, d_emb_mean)  
        
        # 基于候选词（见下）
        
        # 2) sents 嵌入(由句子长度分离)
        sents_len = doc_embs['d_sents_len']
        
        # 用于baseline(第1句平均池化作为doc的表示)
        first_sent_len = sents_len[0]
        d_emb_t_max = np.max(np.array(doc_tokens_embs[:first_sent_len]),axis=0)
        d_emb_t_mean = np.mean(np.array(doc_tokens_embs[:first_sent_len]),axis=0)
        dic['d_emb_t'] = (d_emb_t_max, d_emb_t_mean)
        
        # dic['d_sent_cans_sentidx'] = d_cans_sentidx 
        d_sents_embs_max, d_sents_embs_mean, token2sent_idx = get_sents_embeddings(doc_tokens_embs, sents_len)
        dic['d_sent_embs'] = (d_sents_embs_max, d_sents_embs_mean)

        # 3) cans 嵌入
        doc_tokens = doc_embs['d_tokens']  # 1d
        doc_tokens_pos = doc_embs['d_tokens_pos'] # 1d
        doc_tokens_tags = prepare_word_tags(doc_tokens,doc_tokens_pos) # 构造POS二元组(含处理停用词)
        
        # 3.1) 一次性抽取 (候选+嵌入; 作者方法; bert编码，须注意字符长度限制)
        doc_cans = extract_candidates(doc_tokens_tags) # 2元组列表：['can',(start,end)]; 全部候选！！！
        d_can_embs_max_author, d_can_embs_mean_author = cans_embeddings(doc_tokens_embs, doc_cans) # 方法内进行长度控制
        
        dic['d_emb_embedrank_author'] = doc_embeddings(doc_tokens_embs, doc_cans)
        # Can 8元组: cp, (emb_max,emb_min), c_posi, c_posi_a, c_s_posi, len, c_posi_lst, c_s_posi_lst
        d_can_author_max = extend_cans(doc_cans, d_can_embs_max_author, token2sent_idx)  # 7元组列表
        d_can_author_mean = extend_cans(doc_cans, d_can_embs_mean_author, token2sent_idx)
        dic['d_can_author'] = (d_can_author_max, d_can_author_mean)
        
        # TODO:计算句子表示、基于句子候选抽取(自己+)
        # 2）按句抽取(错误分句时can可能不同)
        d_cans_s, d_cans_embs_s_max, d_cans_embs_s_mean = get_cans_embeddings(doc_tokens_embs, doc_tokens_tags, sents_len)
        
        dic['d_emb_embedrank'] = doc_embeddings(doc_tokens_embs, d_cans_s)
        # Can 8元组:
        d_sent_cans_max = extend_cans(d_cans_s, d_cans_embs_s_max, token2sent_idx) # 与原次序不一致
        d_sent_cans_mean = extend_cans(d_cans_s, d_cans_embs_s_mean, token2sent_idx)
        dic['d_sent_cans'] = (d_sent_cans_max, d_sent_cans_mean)
        
        if len(doc_cans)!=len(d_cans_s):
            diff_idx.append(d_id)
        
        docs_feats.append(dic)  # 每个文档对应一个字典
        
    print(f"一次抽取与按句抽取结果不同:{len(diff_idx)}个,行索引:{diff_idx}")
    
    # 候选短语保存至文件（测试）
    # utils.save_list2txt(cans_lst,os.path.join(config.DATA_SET_DIR,'Inspec','cans.txt'))
    
    return docs_feats
    
if __name__=="__main__":
    ds_name = sys.argv[1]  # 设置1
    # ds_name = 'SemEval2010'
    
    file_path="{}/{}".format(config.DATA_SET_DIR,ds_name)
    model_name=config.EMB_MODEL
    
    docs_embs_kpl_path = os.path.join(file_path, f"test.doclevel.embeddings.{model_name}.pkl")
    with open(docs_embs_kpl_path, 'rb') as f:
        docs_embeddings = pickle.load(f)

    docs_feats_kpl_path = os.path.join(file_path, f"test.alllevel.feats.{model_name}.pkl")
    if not os.path.exists(docs_feats_kpl_path):
        docs_feats = get_sents_cans_embs(docs_embeddings) # 获取句子和短语embeddings
        with open(docs_feats_kpl_path, 'wb') as f:
            pickle.dump(docs_feats, f)