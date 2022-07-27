import torch
from utils.os_helper import mkdir
pkl_path = '/home/cjl/ssd/ywj/material/rgbModel/PPLiteRgbCatSpectral_Hf_500000_0.001_2_1306/150.pkl'
# mkdir('./new_pkl/')
new_pkl_path = '/home/cjl/ssd/ywj/material/rgbModel/PPLiteRgbCatSpectral_Hf_500000_0.001_2_1306/new_150.pkl'
state_dict = torch.load(pkl_path)#xxx.pth或者xxx.pt就是你想改掉的权重文件
torch.save(state_dict, new_pkl_path, _use_new_zipfile_serialization=False)
