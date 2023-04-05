import torch
from torch import nn
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed
from data_embed import decode_train, decode_valid
import tqdm
from tqdm import tqdm
import pdb
import random
from accelerate import Accelerator
import pickle
from transformers import pipeline, set_seed
from data_embed import gene_embedding
from load_gene2seq import load_gene2seq
from deal_pubtator import load_pubtator
#import wandb
print(1)
gene2seqPath = '/data/kaichh/Biotranslater/dataseq2gene/prot_seq2gene.json'
gene2seq = load_gene2seq(gene2seqPath)
print(2)
data_path='/data/kaichh/Biotranslater/datapubtator/bioconcepts2pubtatorcentral.offset'
model_path = '/data/kaichh/Biotranslater/biogpt_model/emb_model/SmallBioModel_epoch:1_item:99_base.pth'

print(0)
decode_train(gene2seqPath, data_path, 10000000000)