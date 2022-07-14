import sys
sys.path.append('DialogRPT/src')
sys.path.append('2021-R3-Baselines/PrLM')
import numpy as np

TRAIN_BS = int(sys.argv[1])
VALID_BS = int(sys.argv[2])
LR = float(sys.argv[3])
EPOCHS = int(sys.argv[4])
PATIENCE = int(sys.argv[5])

print(f'Finetuning DGPT on DD with LR:{LR} and EPOCHS:{EPOCHS}')

from torch.utils.data import Dataset, DataLoader

C_MAX_LEN = 100
R_MAX_LEN = 28

## Model ##
from transformers19 import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch
tok = GPT2Tokenizer.from_pretrained('gpt2')
model_config = GPT2Config(n_embd=1024, n_layer=24, n_head=16)        
model = GPT2LMHeadModel(model_config)
weights = torch.load('restore/medium_ft.pkl')
if "lm_head.decoder.weight" in weights:
    weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
    weights.pop("lm_head.decoder.weight",None)
model.load_state_dict(weights)
tok.add_special_tokens({'cls_token':'__eou__'})
model.resize_token_embeddings(tok.vocab_size+1)

from torch.utils.data import Dataset, DataLoader

C_MAX_LEN = 100
R_MAX_LEN = 28

class Custom_Dataset(Dataset):
  def __init__(self,tsv_file):
    self.data = []

    for l in open(tsv_file,'r'):
      c,s,r = l.split('\t')
      ctx = tok.encode(c+" __eou__ "+s,return_tensors='pt')
      p = ctx.shape[1]
      #ctx = torch.nn.functional.pad(ctx,(0,C_MAX_LEN-ctx.shape[1]),"constant",0)
      d = tok.encode(c+" __eou__ "+s+" __eou__ "+r,return_tensors='pt')
      inp = d[:,max(p-C_MAX_LEN,0):p+R_MAX_LEN]#torch.concat((ctx,r,torch.tensor([[50256]])),dim=1)
      if inp.shape[1]<C_MAX_LEN+R_MAX_LEN+1:
        inp_1 = torch.concat((inp,torch.LongTensor([[50256]]),torch.full((1,C_MAX_LEN+R_MAX_LEN-inp.shape[1]),0,dtype=torch.long)),dim=1)
        lab = torch.concat((inp,torch.LongTensor([[50256]]),torch.full((1,C_MAX_LEN+R_MAX_LEN-inp.shape[1]),-100,dtype=torch.long)),dim=1)
      #r = torch.nn.functional.pad(r,(0,R_MAX_LEN-r.shape[1]),"constant",0)
      lab[:,:min(p,C_MAX_LEN)] = -100
      lab[:,inp.shape[1]+1:] = -100
      mask = torch.zeros((inp_1.shape[1]-1),dtype=torch.long)
      mask[:inp.shape[1]+1] = 1
      self.data.append({'con':ctx.shape[1],'res':d.shape[1]-p-1,'len':inp.shape[1]+1,'mask':mask,'inp':torch.LongTensor(inp_1)[0,:],'label':torch.LongTensor(lab)[0,:]})

  def __getitem__(self,index):
    return self.data[index]
  
  def __len__(self):
    return len(self.data)


def finetune(model,lr,epochs,reset_patience=5):
  
  patience = reset_patience
  optimizer = torch.optim.Adam(model.parameters(),lr=lr)
  model = model.to(device)

  best_loss = np.inf

  for e in range(epochs):
    
    train_loss = train(model,optimizer,e)
    val_loss = validate(model,e)
    
    ## UNCOMMENT BELOW IF WANT TO SAVE AFTER EVERY EPOCH
    # torch.save({
    #         'epoch': e,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'train_loss': train_loss,
    #         'val_loss': val_loss,
    #         }, f'DGPT_ftune_dd_chkpt_{e}.pth.tar')

    if val_loss<best_loss:
      patience = reset_patience
      best_loss = val_loss
      torch.save(model.state_dict(), f'DGPT_ftune_dd_chkpt_best_{e}.pkl')
    else:
      patience-=1
    if patience<0:
      print('Out of patience...terminating training')
      break

def train(model,optimizer,e):
  model.train()

  steps = 0
  agg_loss = 0

  for b in train_dl:
    mask = b['mask'].double()
    loss = model(input_ids=b['inp'].cuda(),labels=b['label'].cuda(),attention_mask=mask.cuda())[0]
    optimizer.zero_grad()
    agg_loss += loss.item()
    loss.backward()
    optimizer.step()
    steps+=1

    if steps%100==0:
      print(f'Epoch: {e}\tSteps: {steps}\tLoss per step: {agg_loss/steps}')
    
  return agg_loss/steps

def validate(model,e):
  model.eval()

  steps = 0
  agg_loss = 0

  for b in valid_dl:
    mask = b['mask'].double()
    loss = model(input_ids=b['inp'].cuda(),labels=b['label'].cuda(),attention_mask=mask.cuda())[0]
    agg_loss += loss.item()
    steps+=1

  print(f'Val Loss per step after {e} epochs: {agg_loss/steps}')
  return agg_loss/steps


train_dataset = Custom_Dataset('new_preprocessed_data/DD/dd_train_input.tsv')
valid_dataset = Custom_Dataset('new_preprocessed_data/DD/dd_valid_input.tsv')

train_dl = DataLoader(train_dataset,batch_size=TRAIN_BS,shuffle=True)
valid_dl = DataLoader(valid_dataset,batch_size=VALID_BS,shuffle=True)




import torch
device = 'cuda:0'

finetune(model,lr=LR,epochs=EPOCHS)
