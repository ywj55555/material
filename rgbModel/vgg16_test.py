import torch
tmp = torch.tensor([[0,0.1,0,0],[0,0,0.2,0]])
out = torch.argmax(tmp,dim=1)
print(out)