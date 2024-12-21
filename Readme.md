# HCUKE
## ranker.py参数
sim_methods['d-s'] = 'cos'
sim_methods['s-c'] = 'cos'
sim_methods['c-c'] = 'cos'

doc_type = 'doc'
poolings['doc_pooling'] = 'max' 
poolings['sent_pooling'] = 'max'
poolings['can_pooling'] = 'max'

posi_type = '1'     
len_type = '0'

# RUN
- Step1: 生成文档嵌入(get_doc_embs.py)，每个数据集1个
  - 运行：get_doc_embs.py Inspec
  - 结果：test.doclevel.embeddings.bert-base-uncased.pkl

- Step2: 抽取候选及其表示(get_sent_ckp_embs.py)，每个数据集2个
  - 运行：get_sent_ckp_embs.py Inspec
    - T,A拼接: D一次抽; 按句抽. 另: T仅2种池化表示
        结果：test.alllevel.feats.bert-base-uncased.pkl

- Step 3: 准备模型计算所需的数据 (build_data.py)
  - 运行: build_data.py Inspec max 0
  - 结果：
  
- Step 4: 排名 (ranker.py) 
  - 运行: ranker.py Inspec mean cls dot