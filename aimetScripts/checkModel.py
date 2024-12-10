import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from aimet_torch.model_validator.model_validator import ModelValidator
from funasr.models.sanm.encoder import MyCustomeModule,SANMEncoder
import funasr.models.paraformer_mobile


# model_file_path = '/data/workspace/projects/cgameAssets/new_binary_pt/model_encoder.pt'
model_file_path = '/data/workspace/projects/cgameAssets/new_binary_pt_1/model_encoder_1.pt'
model = torch.load(model_file_path)

model.eval()

dummy_input = model.export_dummy_inputs()

if device == "cuda":
    model = model.cuda()
    if isinstance(dummy_input,torch.Tensor):
        dummy_input = dummy_input.cuda()
    else:
        dummy_input = tuple([i.cuda] for i in dummy_input)
    
ModelValidator.validate_model(model, dummy_input)

print('pass!')