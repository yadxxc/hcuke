This is code for KBS paper: [HCUKE: A Hierarchical Context-aware approach for Unsupervised Keyphrase Extraction](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011456).

The preprocessed datasets used for the evaluation also comes from [JointGL](https://github.com/xnliang98/uke_ccrank)..

# Citation
If you use this code, please cite our paper:
```
@article{XU2024112511,
title = {HCUKE: A Hierarchical Context-aware approach for Unsupervised Keyphrase Extraction},
journal = {Knowledge-Based Systems},
volume = {304},
pages = {112511},
year = {2024},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2024.112511},
url = {https://www.sciencedirect.com/science/article/pii/S0950705124011456},
author = {Chun Xu and Xian-Ling Mao and Cheng-Xin Xin and Yu-Ming Shang and Tian-Yi Che and Hong-Li Mao and Heyan Huang}
}
```

# Requirments
- transformers==3.0.2
- nltk
- pytorch (conda install pytorch==1.7.1 cudatoolkit=11.0 -c pytorch )
- tqdm

# RUN
- Step1: Generate document embedding => test.doclevel.embeddings.bert-base-uncased.pkl
```step1
python get_doc_embs.py Inspec
```

- Step2: Extract candidates and their representations => test.alllevel.features.bert-base-uncased.pkl
```step2
python get_sent_ckp_embs.py Inspec
```

- Step 3: Prepare data for model calculation => data.Inspec.bert-base-uncased.max.pkl
```step3
python get_sent_ckp_embs.py Inspec
```
  - Run: python build_data.py Inspec max 1
  - Result: 
  
- Step 4: Ranking candidates => DUC2001/Inspec/SemEval2010 (F1@5,F1@10,F1@15)
```step4
python python ranker.py 
```
