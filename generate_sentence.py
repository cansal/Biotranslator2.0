from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed
from model import MyModel
import torch
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = MyModel()

model.load_state_dict (torch.load('biogpt_model/BioModel_epoch:3_item:23610.pth')['model_state_dict'])

model.to('cpu')
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
print("see")
print(generator("CAT is", max_length=20, num_return_sequences=5, do_sample=True))