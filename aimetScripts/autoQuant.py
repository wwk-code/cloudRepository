import os
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset,load_from_disk
from aimet_torch.model_preparer import prepare_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from collections import namedtuple
import torch.nn as nn 
import time
from transformers import AdamW
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quantsim import QuantParams
from aimet_torch.bias_correction import correct_bias
import numpy as np
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.auto_quant import AutoQuant
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
import pdb



Output = namedtuple('Output', ['logits'])

class CustomModel(nn.Module):

    def __init__(self, model_name:str,device: torch.device):
        super(CustomModel, self).__init__()
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(self.device)

    def forward(self, input_ids, attention_mask, label = None):
        if isinstance(input_ids,str):
            input_ids = torch.randint(0,1000, (1, 128)).to(torch.int64).to(self.device)
        if isinstance(attention_mask,str):
            attention_mask = torch.ones((1, 128)).to(torch.int64).to(self.device)
        # 获取模型输出
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # 返回包含 logits 的命名元组
        return Output(logits=logits)

class TextDataset(Dataset):

    def __init__(self, sentences, labels, tokenizer):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.sentences[idx], 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=max_length
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0).to(torch.int64),
            'attention_mask': inputs['attention_mask'].squeeze(0).to(torch.int64),
            'label': torch.tensor(self.labels[idx], dtype=torch.int64)
        }


class ModelDataPipeline:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CustomModel(model_name,self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sentences = raw_sentences
        self.labels = raw_labels
        self.dataset = TextDataset(self.sentences, self.labels, self.tokenizer)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

    def evaluate(self, dummy_inputs=None) -> float:
        self.model.to(self.device)
        self.model.eval()

        val_loader = self.data_loader
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k.to(self.device) : v.to(self.device) for k, v in batch.items() if k != 'label'}
                labels = batch['label'].to(self.device)
                if dummy_inputs != None:
                   inputs = {'input_ids':dummy_inputs[0].to(self.device),'attention_mask':dummy_inputs[1].to(self.device)} 
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        accuracy = correct_predictions / total_predictions
        return accuracy


def eval_callback(model: torch.nn.Module) -> float:
    return pipeline.model.evaluate()


model_name = 'philschmid/tiny-bert-sst2-distilled'
initialDataSetLoadSize = 1024
modelPipelineSampleSize = 128
batch_size = 8
max_length = 128
loadFromCache = True
dataset_tag = 'train'


if not loadFromCache:
    dataset = load_dataset('glue', 'sst2',split=f'test[:{initialDataSetLoadSize}]')
else:
    cache_directory = '/root/.cache/huggingface/datasets/glue/sst2/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/'
    dataset = load_dataset('arrow',data_files=f'{cache_directory}/glue-validation.arrow')

unlabeled_sentences = dataset.map(lambda x: {'sentence': x['sentence']}, remove_columns=['label'])
unlabeled_dataloader = DataLoader(unlabeled_sentences, batch_size=batch_size, shuffle=True)
raw_sentences = [example['sentence'] for example in dataset[dataset_tag]][:modelPipelineSampleSize]
raw_labels = [example['label'] for example in dataset[dataset_tag]][:modelPipelineSampleSize]
use_cuda = torch.cuda.is_available()

pipeline = ModelDataPipeline()

if __name__ == "__main__":
    
    duumy_input_ids = torch.randint(0,1000, (1, 128)).to(torch.int64).to('cuda')
    duumy_token_ids = torch.zeros(1, 128).to(torch.int64).to('cuda')
    duumy_attention_mask = torch.ones((1, 128)).to(torch.int64).to('cuda')
    dummy_input_shape = [(1,128),(1,128)]

    auto_quant = AutoQuant(model=pipeline.model,dummy_input=duumy_input_ids,data_loader=unlabeled_dataloader,eval_callback=eval_callback)
    adaround_params = AdaroundParameters(data_loader=pipeline.data_loader, num_batches=len(pipeline.data_loader), default_num_iterations=32)
    auto_quant.set_adaround_params(adaround_params)
    model, accuracy = auto_quant.run_inference()
    sim, initial_accuracy = auto_quant.run_inference()
    model, optimized_accuracy, encoding_path = auto_quant.optimize(allowed_accuracy_drop=0.01)
    model, accuracy = auto_quant.run_inference()
    print(f"- Quantized Accuracy (before optimization): {initial_accuracy:.4f}")
    print(f"- Quantized Accuracy (after optimization):  {optimized_accuracy:.4f}")

    print('finished!')