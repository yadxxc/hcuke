import os
import json
import pickle as pkl

import pandas as pd
from datetime import datetime
import sys
from nltk.stem import PorterStemmer
import numpy as np

# 返回字典列表
def load_jsonline(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [json.loads(line.strip()) for line in lines]


def save_jsonline(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            json.dump(line, f, ensure_ascii=False)
            f.writelines('\n')
    
    print(f"Data has been saved into {path}.")
    return data

# 返回文本列表
def load_txt(path, sep=None):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip().split(sep) if sep is not None else line.strip() for line in lines]


def save_txt(data, path, sep=None):
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            if type(line) == list and sep is not None:
                line = sep.join(line)
            else:
                line = line.strip()
            f.writelines(line + "\n")
    print(f"Data has been saved into {path}.")
    return None

def save_list2txt(data, path, sep=";"):
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            if type(line) == list and sep is not None:
                line = sep.join(line)
            else:
                line = line.strip()
            f.writelines(line + "\n")
    print(f"Data has been saved into {path}.")
    return None

def save_pkl(data, path):
    with open(path, 'wb') as f:
        pkl.dump(data, f)
    print(f"Data has been saved into {path}.")
    return None

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

def make_dirs(path):
    if not os.path.exists(path):
        print(f"{path} does not exist, create it.")
        os.makedirs(path)
    return None

# 时间戳文件名
def get_filename_with_datetime(main_file_name, ext_file_name):
    now = datetime.now()
    filename = now.strftime(f"{main_file_name}_%Y-%m-%d_%H-%M-%S.{ext_file_name}")
    return filename

def list2excel(res_2d_list,out_dir=None):
    # 无表头
    df = pd.DataFrame(res_2d_list,)
    df.to_csv('f1_results_{}.csv'.format(datetime.now().strftime('%Y-%m-%d_%H%M%S')))
    df.to_excel('f1_results_{}.xlsx'.format(datetime.now().strftime('%Y-%m-%d_%H%M%S')))

def result2excel(res_2d_list,out_dir):
    # csv表头
    col_names = ['dataset', 'F1@5', 'F1@10', 'F1@15', 'hyper_p']
    df = pd.DataFrame(res_2d_list, columns=col_names)
    df.to_csv('f1_results_{}.csv'.format(datetime.now().strftime('%Y-%m-%d_%H%M%S')))
    df.to_excel('f1_results_{}.xlsx'.format(datetime.now().strftime('%Y-%m-%d_%H%M%S')))
    
def result2exceldetails(res_2d_list,out_dir):
    # csv表头
    col_names = ['dataset', 'F1@5', 'P@5', 'R@5','F1@10', 'P@10', 'R@10', 'F1@15', 'P@15', 'R@15', 'hyper_p']
    df = pd.DataFrame(res_2d_list, columns=col_names)
    df.to_csv('all_results_{}.csv'.format(datetime.now().strftime('%Y-%m-%d_%H%M%S')))
    df.to_excel('all_results_{}.xlsx'.format(datetime.now().strftime('%Y-%m-%d_%H%M%S')))

# 公共1：向量-向量(点乘dot、曼哈顿man、余弦cos、欧式euc等)
def vec_vec_dist(vec1, vec2, method='dot'):
    assert len(vec1) == len(vec2), "len(vec1) != len(vec2)"
    vec1 = np.array(vec1) if isinstance(vec1, list) else vec1
    vec2 = np.array(vec2) if isinstance(vec2, list) else vec2

    # 越大越不相似: euc,man 相反：cos,dot
    if method == "euc":  
        dist = np.linalg.norm(vec1 - vec2)  
        # sim = np.sqrt(np.sum((vec1-vec2)**2)) # 直接
    elif method == "man": 
        # sim = 1 / np.linalg.norm(vec1-vec2,ord=1) # 越大越不相似,需取倒数
        d_abs = np.sum(np.abs(vec1 - vec2))  # 差向量每个元素先取绝对值，在求和
        dist = d_abs if d_abs != 0 else sys.float_info.epsilon  # 防止除数为0
    elif method == "cos":
        dist = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))  # -- 越大越相似
    else:  # dot
        dist = vec1.dot(vec2)  # -- 越大越相似
    return dist

def vec_vec_sim(vec1, vec2, method='dot'):
    assert len(vec1) == len(vec2), "len(vec1) != len(vec2)"
    vec1 = np.array(vec1) if isinstance(vec1, list) else vec1
    vec2 = np.array(vec2) if isinstance(vec2, list) else vec2
    
    res_array = vec_vec_dist(vec1, vec2, method)
    if method in ['cos', 'dot']: # 越大越相似
        res_array = res_array
    else:                        # 越小越相似, 'man','euc'
        res_array = 1 / (res_array + sys.float_info.epsilon)
    return res_array

# 公共2：1d数组 规范化:max-min; 0-mean; 标准差(或Z-score)
# Y=(x-min)/(max-min); Y=x-mean; (x-μ)/σ
def normalization(array_1d, method='max_min'):
    mean_v = np.mean(array_1d)
    if method == 'max_min':
        max_v = np.max(array_1d)
        min_v = np.min(array_1d)
        interval = max_v - min_v if max_v - min_v != 0 else sys.float_info.epsilon
        res = (array_1d - min_v) / interval
    elif method == '0_mean':
        res = array_1d - mean_v
    elif method == 'softmax':
        res = np.exp(array_1d) / np.sum(np.exp(array_1d))
    else:
        res = (array_1d - mean_v) / np.std(array_1d)
    return res


def simple_word_overlap_similarity(expression1, expression2, isstem=True, islower=True, lambda1 = 0):
    """
    Computes the overlap similarity between two single or multi-word
    expressions.

    @param    expression1: The first single or multi-word expression.
    @type     expression1: C{string}
    @param    expression2: The second single or multi-word expression.
    @type     expression2: C{string}

    @return:  The overlap similarity computed between the two expressions.
    @rtype:   C{float}
    """
    expression1 = expression1.lower() if islower else expression1
    expression2 = expression2.lower() if islower else expression2

    words1 = expression1.split()
    words2 = expression2.split()

    ps =PorterStemmer()
    words1 = [ps.stem(w.strip()) for w in words1] if isstem else words1
    words2 = [ps.stem(w.strip()) for w in words2] if isstem else words2

    intersection = set(words1) & set(words2)
    union = set(words1) | set(words2)
    sim = float(len(intersection)) / float(len(union))
    sim = sim if sim >= lambda1 else 0    # lambda1的经验值为0.25
    return sim

def issname(expression1, expression2, isstem=True, islower=True):

    expression1 = expression1.lower() if islower else expression1
    expression2 = expression2.lower() if islower else expression2

    words1 = expression1.split()
    words2 = expression2.split()

    ps =PorterStemmer()
    words1 = [ps.stem(w.strip()) for w in words1] if isstem else words1
    words2 = [ps.stem(w.strip()) for w in words2] if isstem else words2

    expression1 = " ".join(words1)
    expression2 = " ".join(words2)

    return expression1 == expression2

# if __name__=="__main__":
#     print(issname("my machine","My Machine",True,False))