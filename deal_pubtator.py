import json
from tqdm import tqdm
import numpy as np

#filename: /data/kaichh/Biotranslater/datapubtator/bioconcepts2pubtatorcentral.offset
def load_pubtator(filename, gene2seq, artnum):
    obj_list = []
    obj = None
    with open(filename, 'r') as f:
        total_article = 0
        for line in f:
            #with open("sample.txt", "a") as file:
            #    file.write(line)
            
            line = line.strip()
            if not line:
                continue
            
            if '|t|' in line[:30]:
                total_article = total_article + 1
                if(total_article%100 ==0):
                    print("readarticle:{}".format(total_article))

                if total_article == artnum:
                    obj = None
                    break
                if (obj !=None):
                    obj_list.append(obj)
                    obj = None
                obj = {}
                obj['id'] = line.split("|t|")[0]
                obj['title'] = line.split("|t|")[1]
                obj['gene']=[]
                obj['abstract']=['IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII']
                
            elif '|a|' in line[:30]:
                obj['abstract'] = line.split("|a|")[1]
                
            else:
                segment = line.replace(" ", "").split("\t")

                if segment[4]=="Gene":
                    if segment[3] in gene2seq:
                        if segment[3] in obj['abstract']:
                            if segment[3] not in obj['gene']:
                                obj['gene'].append(segment[3])
        if obj != None:
            obj_list.append(obj)
    return obj_list
            
                    
            
