from load_gene2seq import load_gene2seq
from deal_pubtator import load_pubtator
import numpy as np
import torch
from torch.utils.data import Dataset

gene2seq = load_gene2seq('/data/kaichh/Biotranslater/dataseq2gene/prot_seq2gene.json')

#objList = load_pubtator('/data/kaichh/Biotranslater/datapubtator/bioconcepts2pubtatorcentral.offset', gene2seq, 10000)
def gene_embedding(seq, start=0, max_len=2000):
    '''
    One-Hot encodings of protein sequences,
    this function was copied from DeepGOPlus paper
    :param seq:
    :param start:
    :param max_len:
    :return:
    '''
    AALETTER = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    AAINDEX = dict()
    for i in range(len(AALETTER)):
        AAINDEX[AALETTER[i]] = i + 1
    onehot = np.zeros((21, max_len), dtype=np.int32)
    l = min(max_len, len(seq))
    for i in range(start, start + l):
        onehot[AAINDEX.get(seq[i - start], 0), i] = 1
    onehot[0, 0:start] = 1
    onehot[0, start + l:] = 1
    return onehot

from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

trainloader = []
'''
print(tokenizer.encode(objList[0]['abstract']))
    
for l in objList[0]['gene']:
    print(l+":")
    print(tokenizer.encode(l))

for obj in objList:
    id = obj['id']
    abstract = obj['abstract']
    abstractToken = tokenizer.encode(obj['abstract'])
'''
from tqdm import tqdm

import hashlib

def hash_to_01(input_string):
    sha256 = hashlib.sha256(input_string.encode())
    hash_value = sha256.digest()[-1]  # 取最低位
    return hash_value % 5



def decode_train(gene2seqPath, pubtatorPath, artnum):
    gene2seq = load_gene2seq(gene2seqPath)
    objList = load_pubtator(pubtatorPath, gene2seq, artnum)
    examples = []
    for obj in tqdm(objList[18130026:], desc='Processing data'):
        #print("1")
        if obj['gene']==[]:
            continue 
        #print("2")
        abstractTokens = tokenizer.encode(obj['abstract'],max_length=200+len(tokenizer.encode(obj['abstract'])), padding='max_length')#this is like a shit and need to change
        #print("3")
        
        for gene in obj['gene']:
            if(hash_to_01(gene)==0):
                #print("fbiwarningi")
                #print("skip:{}".format(gene))
                continue
            #print("4")
            geneSeqTokens = gene_embedding(gene2seq[gene])
            geneName = tokenizer.encode(gene)
            geneNameTokens = tokenizer.encode(gene, max_length = 10, padding = 'max_length')
            #print("5")da
            #for i in range(2, min(len(abstractTokens)-1, 999)):
            #self.examples.append((abstractTokens[:100], geneNameTokens, geneSeqTokens, abstractTokens[100]))
            i = 0
            while(i+len(geneName)<len(abstractTokens)-5 and abstractTokens[i]!=1):
                if(abstractTokens[i+1:i+len(geneName)]==geneName[1:]):
                    k = i
                    while(abstractTokens[k]!=4 and abstractTokens[k]!=20073 and abstractTokens[k]!= 927 and k>0):
                        k = k-1
                    e = i+1
                    while(abstractTokens[e]!=4 and abstractTokens[e]!=20073 and abstractTokens[e]!= 927 and abstractTokens[e]!=1 and e>0):
                        e = e+1
                    e = e+1
                    ab = torch.cat([torch.tensor([2]),torch.tensor(abstractTokens[k+1:min(k+100,e)]),torch.ones(k+100-min(k+100,e))], dim = 0)
                    examples.append((ab, geneNameTokens, geneSeqTokens, abstractTokens[k+100]))
                    with open("train_dataset.txt",'a') as f:
                        f.write(tokenizer.decode(ab)+"\n")
                        f.write(gene+"\n")
                    #print(tokenizer.decode(torch.cat([torch.tensor([2]),torch.tensor(abstractTokens[k+1:k+100])], dim = 0)))
                    break
                i = i+1
                
    
        #for k in range(20):
            #       if abstractTokens[100*k]==1:
        #            break
            #       if(geneName[1:] in abstractTokens[100*k:100*(k+1)]):
            #         print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
            #        self.examples.append((abstractTokens[100*k:100*(k+1)], geneNameTokens, geneSeqTokens, abstractTokens[100*(k+1)]))
            #    if abstractTokens[100*k+50]==1:
        #            break
            #    if(geneName[1:] in abstractTokens[100*k+50:100*(k+1)+50]):
            #         self.examples.append((abstractTokens[100*k+50:100*(k+1)+50], geneNameTokens, geneSeqTokens, abstractTokens[100*(k+1)+50]))
        
            #print("6")
   
def decode_valid(gene2seqPath, pubtatorPath, artnum):
    gene2seq = load_gene2seq(gene2seqPath)
    objList = load_pubtator(pubtatorPath, gene2seq, artnum)
    examples = []
    for obj in tqdm(objList[18130027:], desc='Processing data'):
        #print("1")
        if obj['gene']==[]:
            continue 
        #print("2")
        abstractTokens = tokenizer.encode(obj['abstract'],max_length=200+len(tokenizer.encode(obj['abstract'])), padding='max_length')#this is like a shit and need to change
        #print("3")
        
        for gene in obj['gene']:
            if(hash_to_01(gene)!=0):
                #print("fbiwarningi")
                #print("skip:{}".format(gene))
                continue
            #print("4")
            geneSeqTokens = gene_embedding(gene2seq[gene])
            geneName = tokenizer.encode(gene)
            geneNameTokens = tokenizer.encode(gene, max_length = 10, padding = 'max_length')
            #print("5")da
            #for i in range(2, min(len(abstractTokens)-1, 999)):
            #self.examples.append((abstractTokens[:100], geneNameTokens, geneSeqTokens, abstractTokens[100]))
            i = 0
            while(i+len(geneName)<len(abstractTokens)-5 and abstractTokens[i]!=1):
                if(abstractTokens[i+1:i+len(geneName)]==geneName[1:]):
                    k = i
                    while(abstractTokens[k]!=4 and abstractTokens[k]!=20073 and abstractTokens[k]!= 927 and k>0):
                        k = k-1
                    e = i+1
                    while(abstractTokens[e]!=4 and abstractTokens[e]!=20073 and abstractTokens[e]!= 927 and abstractTokens[e]!=1 and e>0):
                        e = e+1
                    e = e+1
                    ab = torch.cat([torch.tensor([2]),torch.tensor(abstractTokens[k+1:min(k+100,e)]),torch.ones(k+100-min(k+100,e))], dim = 0)
                    examples.append((ab, geneNameTokens, geneSeqTokens, abstractTokens[k+100]))
                    with open("valid_dataset.txt",'a') as f:
                        f.write(tokenizer.decode(ab)+"\n")
                        f.write(gene+"\n")
                    #print(tokenizer.decode(torch.cat([torch.tensor([2]),torch.tensor(abstractTokens[k+1:k+100])], dim = 0)))
                    break
                i = i+1
                
    
class MyDataset(Dataset):
    def __init__(self, tokenizer, dataPath, gene2seqPath):
        self.gene2seq = load_gene2seq(gene2seqPath)
        self.examples = []
        self.tokenizer = tokenizer
        with open(dataPath, 'r') as f:
            #print(1)
            while(1==1):
                abs = f.readline()
                
                if not abs:
                    #print(2)
                    
                    break
                if abs =="\n":
                    continue
                #print(3)
                gene = f.readline()
                
                abs = abs.replace("\n","")
                gene = gene.replace("\n","")
                
                #print("abs:{}".format(abs))
                #print("gene:{}".format(gene))
                
                self.examples.append((abs, gene))
                #print(4)
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        abs, gene = self.examples[index]

        
        abstractTokens = self.tokenizer.encode(abs, max_length=110, padding='max_length')
        labels = abstractTokens[100]
        
        abstractTokens = abstractTokens[:100]
        abstractTokens = torch.tensor(abstractTokens)
        
        geneNameTokens = tokenizer.encode(gene, max_length = 10, padding = 'max_length')
        geneNameTokens = torch.tensor(geneNameTokens)
        
        geneSeqTokens = gene_embedding(self.gene2seq[gene])
        geneSeqTokens = torch.tensor(geneSeqTokens)
        
        
        labels = torch.tensor(labels)
        #print(tokenizer.decode(abstractTokens.view(-1)))
        return abstractTokens, geneNameTokens, geneSeqTokens, labels
    
    
