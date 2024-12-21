import os

DATA_SET_DIR = os.path.join(os.getcwd(),'data')
EMB_MODEL = "bert-base-uncased"
DATA_SETS = ['DUC2001','Inspec','SemEval2010']
# def comm_opts(parser):
#     # Dictionary Options
#     parser.add_argument('-data_path', type=str, default='/home/xc/kpe2023/uke_ccrank2021/data/',
#                         help="base path of data")
#     parser.add_argument('-model_name', type=str, default='bert-base-uncased',
#                         help="model name for embeddings")
#     parser.add_argument('-doc_embs_file_name', type=str, default='',
#                         help="file name of doc-level embeddings")
#     parser.add_argument('-sent_can_embs_file_name', type=str, default='',
#                         help="file name of sen-can-level embeddings")


# def predict_opts(parser):
#     parser.add_argument('-pred_file_path', type=str, required=True,
#                         help="Path of the prediction file.")
#     parser.add_argument('-present_ks', nargs='+', default=['5', '10', '15'], type=str, help='')