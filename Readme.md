This is code for KBS paper: [HCUKE: A Hierarchical Context-aware approach for Unsupervised Keyphrase Extraction](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011456).

## Requirments
- transformers==3.0.2
- nltk
- pytorch (conda install pytorch==1.7.1 cudatoolkit=11.0 -c pytorch )
- tqdm

# RUN
- Step1: Generate document embedding
  - Run: python get_doc_embs.py Inspec
  - Result: test.doclevel.embeddings.bert-base-uncased.pkl

- Step2: Extract candidates and their representations
  - Run: python get_sent_ckp_embs.py Inspec
  - Result: test.alllevel.features.bert-base-uncased.pkl

- Step 3: Prepare data for model calculation
  - Run: python build_data.py Inspec max 0
  - Result: data.Inspec.bert-base-uncased.max.pkl
  
- Step 4: Ranking (ranker.py) 
  - Run: python ranker.py
  - DUC2001(F1@5,F1@10,F1@15) Inspec(F1@5,F1@10,F1@15) SemEval2010(F1@5,F1@10,F1@15) 
