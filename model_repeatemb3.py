import torch
from torch import nn
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed
from data_embed import MyDataset
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
#wandb.init(project='BioTranslater', entity='hkc')

gene2seqPath = '/data/kaichh/Biotranslater/dataseq2gene/prot_seq2gene.json'
gene2seq = load_gene2seq(gene2seqPath)
data_path='/data/kaichh/Biotranslater/datapubtator/bioconcepts2pubtatorcentral.offset'
model_path = '/data/kaichh/Biotranslater/biogpt_model/emb_model/TinyBioModel_epoch:2_item:0_base.pth'
repeat_emb=1
IsSaveData = 0
IsLoadData = 0
IsTrain = 1
IsValid = 1
IsLoad = 0
LoadforTrain = 0
epochs = 200
batch_size = 1
sent = 1000000000
from accelerate import DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")


def generates(model, abstract, maxLen, geneName):
    abstractTokens = torch.tensor(tokenizer.encode(abstract)).view(1,-1)

    for i in range(maxLen-(abstractTokens.size(1))-1):
        geneNameTokens = torch.tensor(tokenizer.encode(geneName)).view(1,-1)   
        geneSeqTokens = torch.tensor(gene_embedding(gene2seq[geneName])).unsqueeze(0)
        labels = torch.tensor([1]).view(1)
        abstractTokens, geneNameTokens, geneSeqTokens, labels = accelerator.prepare(abstractTokens, geneNameTokens, geneSeqTokens, labels)
        bioOut, _ = model(abstractTokens.to(model.device), geneNameTokens.to(model.device), geneSeqTokens.to(model.device), labels.to(model.device))
        max_id = bioOut.max(dim = -1)[1][0][-1]
        max_id = accelerator.prepare(max_id)
        #print(tokenizer.decode(torch.cat([abstractTokens, max_id.view(1, -1).to(abstractTokens.device)], dim = 1).view(-1)))
        abstractTokens = torch.cat([abstractTokens, max_id.view(1, -1).to(abstractTokens.device)], dim = 1).view(1,-1)
    
    return abstractTokens.view(-1)

class CNN(nn.Module):
    
    def __init__(self,
                 hidden_dim=1000,
                 seq_input_nc=21,
                 seq_in_nc=512,
                 seq_max_kernels=129,
                 seq_length=2000,
                 network_dim=800,
                 description_dim=768,
                 text_dim=1024):
        """

        :param seq_input_nc:
        :param seq_in_nc:
        :param seq_max_kernels:
        :param dense_num:
        :param seq_length:
        """
        super(CNN, self).__init__()

        self.para_conv, self.para_pooling = [], []
        kernels = range(8, seq_max_kernels, 8)
        self.kernel_num = len(kernels)
        for i in range(len(kernels)):
            exec("self.conv1d_{} = nn.Conv1d(in_channels=seq_input_nc, out_channels=seq_in_nc, kernel_size=kernels[i], padding=0, stride=1)".format(i))
            exec("self.pool1d_{} = nn.MaxPool1d(kernel_size=seq_length - kernels[i] + 1, stride=1)".format(i))
        self.fc_seq =[nn.Linear(len(kernels)*seq_in_nc, hidden_dim), nn.LeakyReLU(inplace=True)]
        self.fc_seq = nn.Sequential(*self.fc_seq)
        self.cat2emb = nn.Linear(hidden_dim, text_dim)

    def forward(self, x=None, x_description=None, x_vector=None):
        x_list = []
        # 21*2000->
        for i in range(self.kernel_num):
            exec("x_i = self.conv1d_{}(x)".format(i))
            exec("x_i = self.pool1d_{}(x_i)".format(i))
            exec("x_list.append(torch.squeeze(x_i).reshape([x.size(0), -1]))")
        y = self.fc_seq(torch.cat(tuple(x_list), dim=1))

        #x_enc = torch.nn.functional.normalize(x_cat, p=2, dim=1)
        return self.cat2emb(y)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        
    def forward(self, x):
        out = self.fc1(x.view(x.size(0), -1))
        out = self.relu(out)
        out = self.fc2(out)
        return out
torch.autograd.set_detect_anomaly(True)
class MyModel(nn.Module):
    def __init__(self):

        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=42384, embedding_dim=1024)
        self.gpt = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
        self.mlp1 = CNN()
        #self.mlp1 = MLP(42000, 2000, 1024)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self, abstractTokens=None, geneNameTokens=None, geneSeqTokens=None, labels = None):
        #print("here?")
        #print("1")d
        geneNameTokensSize = geneNameTokens.size(1)
        abstractEmb = self.embedding(abstractTokens)
        #print("here??")
        geneSeqEmbs = self.mlp1(geneSeqTokens.float())
        processedTensor = abstractEmb
        processedTokens = abstractTokens
        #print("here???")
        i = 0
        # replacement
        processedTokensjj = torch.zeros(processedTokens.shape)
        processedTensorjj = torch.zeros(processedTensor.shape)
        #print("2")
        for j in range(processedTensor.size(0)):
            processedTokensj = processedTokens[j:j+1,:]
            processedTensorj = processedTensor[j:j+1,:,:]

            geneSeqEmbsj = geneSeqEmbs[j:j+1,:].unsqueeze(dim = 1)

            geneNameTokensSize=geneNameTokens.size(1)
            for r in range(geneNameTokens.size(1)):
                if geneNameTokens[j,r]==1:
                    geneNameTokensSize = r
                    break
            geneNameTokensj = geneNameTokens[j:j+1,:geneNameTokensSize]
            for i in range(-1, processedTensor.size(1)-geneNameTokensSize+1):
                #print(i)
                if (processedTokensj[0, i+1:i+geneNameTokensSize].to(torch.long)).equal(geneNameTokensj[0, 1:].to(torch.long)):
                    #pdb.set_trace()
                    #print(processedTokens.shape)
                    print(tokenizer.decode(geneNameTokensj.view(-1)))

                    processedTensorB = processedTensorj[:, :i+1, :]
                    processedTensorE = processedTensorj[:, i+geneNameTokensSize:, :]
                    processedTensorj = processedTensorB
                    for r in range(geneNameTokensSize-1):
                        processedTensorj = torch.cat([processedTensorj, geneSeqEmbsj.to(processedTensorj.device)], dim = 1)
                    processedTensorj = torch.cat([processedTensorj, processedTensorE], dim = 1)
                    
                    break;
                    '''
                    else:
                        #print("here")
                        #print(tokenizer.decode(geneNameTokensj.view(-1)))
                        
                        processedTokensj = torch.cat([processedTokensj[:, :i+1], torch.tensor(random.randint(0, 42380)).view(1,-1).to(processedTokensj.device), processedTokensj[:, i+geneNameTokensSize:]], dim = 1)
                        processedTensorj = torch.cat([processedTensorj[:, :i+1, :], geneSeqEmbsj.to(processedTensorj.device), processedTensorj[:, i+geneNameTokensSize:, :]], dim = 1)
                        i = 0
                        break;'''
                
            
            processedTensorjj[j:j+1,:,:] = processedTensorj
            processedTokensjj[j:j+1,:] = processedTokensj
        #print("3")
        bioOut = self.gpt(inputs_embeds = processedTensorjj.to(processedTensor.device))[0]
        l = torch.cat([processedTokensjj[:, 1:], torch.tensor(labels.view(processedTokensjj.size(0), -1)).to(processedTokensjj.device)], dim = 1).clone().detach()
        #让out接近词的embedding
       # output = self.mlp2(bioOut)
        
        loss = self.criterion(bioOut.transpose(1,2), l.to(torch.long).to(bioOut.device))
        #print("4")
        return bioOut, loss
print("0")
model = MyModel()

#print(list(model.parameters())[389])
#print(list(model.parameters())[390])
#print(list(model.parameters())[391])
#print(list(model.parameters())[392])

from torch.utils.data import DataLoader, RandomSampler
#
if(IsTrain):


    #print(1)

    if (not IsLoadData):
        print("loading dataset...")
        train_dataset = MyDataset(tokenizer, '/data/kaichh/Biotranslater/toydatatocnn.txt',gene2seqPath)

    



        print("finish loading")
        train_sampler = RandomSampler(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            num_workers=1,
            drop_last=True,
            shuffle= False,
        )    
    if (IsSaveData):
        print("doing...")
        with open('train_data.pkl', 'wb') as f:
            pickle.dump(train_loader, f)
        print("FINISH SAVED")

    if(IsLoadData):
        print("isloading...")
        with open('train_data.pkl', 'rb') as f:
            train_loader = pickle.load(f)
        print("done")
    import torch.optim as optim
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    model, optimizer, train_loader= accelerator.prepare(model, optimizer, train_loader)
    
    if(LoadforTrain):
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        #optimizer.load_state_dict(torch.load(model_path)['optimizer_state_dict'])
    #print("3")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(enumerate(train_loader), desc = "trainning", total = len(train_loader))
        for i, batch in pbar:
           
            #print("epoch:{},training:{}%, total_trainloader:{}".format(epoch, 100*i/len(train_loader), len(train_loader)))
            abstractTokens, geneNameTokens, geneSeqTokens, labels = batch
            abstractTokens = abstractTokens.to(model.device)
            geneNameTokens = geneNameTokens.to(model.device)
            geneSeqTokens = geneSeqTokens.to(model.device)
            labels = labels.to(model.device)
            

            
            output, loss = model(abstractTokens, geneNameTokens, geneSeqTokens, labels)#(1,68),(1,2),(1,1,2000),(1)
            #print("outputs")
            #wandb.log({'train_loss': loss})
            print("train_loss:{}".format(loss))
            #print(outputs)
            #print("output0")
            #print(outputs[0])
            #print('5')
            optimizer.zero_grad()
            #print("6")
            accelerator.backward(loss)
            #print("7")
            optimizer.step()
            
            #print("8")
            epoch_loss += loss.detach().item()

            pbar.set_description(f'Epoch={epoch}, Batch={i+1}, train_loss={loss}')
            with open ("log", 'a') as f:
                f.write(f'Epoch={epoch}, Batch={i+1}, train_loss={loss}'+'\n')
            #print("9")
            if (i %1000 ==0 or i==len(train_loader)-2):
                with open("toymodel.txt", "a")as f:
                    f.write("{} in epoch {} \n".format(i,epoch))
                    f.write(tokenizer.decode(generates(model, "The ten'th letter of NT5E is",20,"NT5E"))+"\n") #t
                    f.write(tokenizer.decode(generates(model, "The ten'th letter of ABHD17A is",20,"ABHD17A"))+"\n") #c
                    
                    f.write(tokenizer.decode(generates(model, "The ten'th letter of FAM50A",20,"FAM50A"))+"\n")#a

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 'biogpt_model/emb_model/ToyBioModel_epoch:{}_item:{}_base.pth'.format(epoch, i)) 
            #if (i % 100000 == 99999 and epoch%1==0):
            #    print(tokenizer.decode(generates(model, "Gene CAT is",50,"CAT")))
            #    print("-----------------------------")
            #    print(tokenizer.decode(generates(model, "Gene CORT is",50,"CORT")))
            #    print("-----------------------------")
            #    print(tokenizer.decode(generates(model, "Gene CORO6 is",50,"CORO6"))) 
             #   print("-----------------------------")
            #    print(tokenizer.decode(generates(model, "Gene COX4I1 is",50,"COX4I1")))
             #   print("-----------------------------")
             #   print(tokenizer.decode(generates(model, "Gene COX5A is",50,"COX5A"))) 
            #print("10")
            #print("10")
#

import nltk.translate.bleu_score as bleu

    

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")



if(IsLoad):
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaa")
    model.load_state_dict(torch.load(model_path)['model_state_dict'])

'''
print(tokenizer.decode(generates(model, "god, Gene CAT is a",50,"CAT"))) 

print("-----------------------------")
print(tokenizer.decode(generates(model, "Gene COX5A is a",50,"COX5A"))) 

print("-----------------------------")
print(tokenizer.decode(generates(model, "COX5A is a",50,"COX5A"))) 

print("-----------------------------")
print(tokenizer.decode(generates(model, "COX5A is",50,"COX5A"))) 

print("-----------------------------")
print(tokenizer.decode(generates(model, "COX5A",50,"COX5A"))) 
'''
if(IsValid):
    valid_dataset = MyDataset(tokenizer, '/data/kaichh/Biotranslater/valid_dataset.txt',gene2seqPath)
    test_sampler = RandomSampler(valid_dataset)
    test_loader = DataLoader(
    valid_dataset,
    sampler=test_sampler,
    batch_size=1,
    num_workers=1,
    drop_last=True,
    shuffle= False,
    )
    score = 0
    for i, batch in tqdm(enumerate(test_loader), desc = "validing"):
        

        abstractTokens, geneNameTokens, geneSeqTokens, labels, gene = batch
        abstractTokens = abstractTokens.to(model.device)
        geneNameTokens = geneNameTokens.to(model.device)
        geneSeqTokens = geneSeqTokens.to(model.device)
        labels = labels.to(model.device)
        gene = gene[0]
        output, loss = model(abstractTokens, geneNameTokens, geneSeqTokens, labels)#
        score+=loss.item()
        print("loss:{}".format(score/(i+1)))
        '''
        result_sentence = generates(model, 'Oh my god, Gene {}'.format(gene), 101, gene)
        print(tokenizer.decode(result_sentence))
        abstra = (abstractTokens.view(-1))
        res = (result_sentence.view(-1))
        bleu_socre = bleu_socre+bleu.sentence_bleu([[int(i) for i in res]], [int(i) for i in abstra])
        print
        

        print("bleu_score:{}".format(bleu_socre/(i+1)))
        '''

