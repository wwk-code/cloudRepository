import os
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
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


Output = namedtuple('Output', ['logits'])


class CustomModel(nn.Module):
    def __init__(self, model_name):
        super(CustomModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).cuda()

    def forward(self, input_ids, attention_mask, label = None):
        if isinstance(input_ids,str):
            input_ids = torch.randint(0,1000, (1, 128)).to(torch.int64).to('cuda')
        if isinstance(attention_mask,str):
            attention_mask = torch.ones((1, 128)).to(torch.int64).to('cuda')
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
            max_length=128
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0).to(torch.int64),
            'attention_mask': inputs['attention_mask'].squeeze(0).to(torch.int64),
            'label': torch.tensor(self.labels[idx], dtype=torch.int64)
        }

class ModelDataPipeline:

    def __init__(self, model_name='philschmid/tiny-bert-sst2-distilled'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CustomModel(model_name)
        # _ = fold_all_batch_norms(self.model, input_shapes=[(1, 128),(1, 128)])
        print('')

    def get_val_dataloader(self, sentences, labels, batch_size=8) -> DataLoader:
        dataset = TextDataset(sentences, labels, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return data_loader

    def evaluate(self, sentences, labels, use_cuda: bool,dummy_inputs = None) -> float:
        self.model.to(self.device)
        self.model.eval()
        val_loader = self.get_val_dataloader(sentences, labels)
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.cuda() for k, v in batch.items() if k != 'label'} if use_cuda else batch
                labels = batch['label'].cuda() if use_cuda else batch['label']
                if dummy_inputs != None:
                   inputs = {'input_ids':dummy_inputs[0].to('cuda'),'attention_mask':dummy_inputs[1].to('cuda')} 
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        accuracy = correct_predictions / total_predictions
        return accuracy
    
    def evaluate_simModel(self, sentences, labels, use_cuda: bool,sim_model) -> float:
        val_loader = self.get_val_dataloader(sentences, labels)
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.cuda() for k, v in batch.items() if k != 'label'} if use_cuda else batch
                labels = batch['label'].cuda() if use_cuda else batch['label']
                outputs = sim_model.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        accuracy = correct_predictions / total_predictions
        return accuracy


    def quantize(self):
        duumy_input_ids = torch.randint(0, self.tokenizer.vocab_size, (1, 128)).to(self.device)
        duumy_attention_mask = torch.ones((1, 128)).to(self.device)
        sim = QuantizationSimModel(model=self.model,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,   # CLE + Bias Correction
                           dummy_input=(duumy_input_ids,duumy_attention_mask),
                           default_output_bw=bitwidth,
                           default_param_bw=bitwidth)
        return sim


    def finetune(self,sim_model,train_sentences, train_labels, epochs=10, learning_rate=2e-5, batch_size=8):
        print('start qat')
        train_loader = self.get_val_dataloader(train_sentences, train_labels, batch_size=batch_size)
        optimizer = AdamW(sim_model.model.parameters(), lr=learning_rate)

        sim_model.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = sim_model.model(**inputs)
                loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')


def pass_calibration_data(sim_model, args):
    pipeline,sentences, labels = args
    data_loader = pipeline.get_val_dataloader(sentences, labels)
    batch_size = data_loader.batch_size
    sim_model.eval()
    batch_cntr = 0
    with torch.no_grad():
        for batch in data_loader:
            for i in range(batch_size):
                # 提取单个样本
                input_data = {k: v[i].unsqueeze(0) for k, v in batch.items() if k != 'label'}
                input_data = {k: v.to(pipeline.device) for k, v in input_data.items()}
                # 执行推理
                sim_model(**input_data)
                batch_cntr += 1
                if (batch_cntr * batch_size) > sample_size:
                    break

def eval_callback(model) -> float:
    return pipeline.model.evaluate(sentences,labels,use_cuda)


# sample_size = 10000
sample_size = 100
bitwidth = 8
EVAL_DATASET_SIZE = sample_size
CALIBRATION_DATASET_SIZE = 20
BATCH_SIZE = 10
dataset = load_dataset('glue', 'sst2')['train']
sentences = dataset.map(lambda x: {'sentence': x['sentence']}, remove_columns=['label'])
unlabeled_dataloader = DataLoader(sentences, batch_size=32, shuffle=True)
# dataset = load_dataset('glue', 'sst2')['validation']
sentences = [example['sentence'] for example in dataset][:sample_size]
labels = [example['label'] for example in dataset][:sample_size]
use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    
    pipeline = ModelDataPipeline()
    iter = 1
    dataloader = pipeline.get_val_dataloader(sentences=sentences,labels=labels)
    duumy_input_ids = torch.randint(0,1000, (1, 128)).to(torch.int64).to('cuda')
    duumy_token_ids = torch.zeros(1, 128).to(torch.int64).to('cuda')
    duumy_attention_mask = torch.ones((1, 128)).to(torch.int64).to('cuda')
    dummy_input_shape = [(1,128),(1,128)]
  
    auto_quant = AutoQuant(model=pipeline.model,dummy_input=duumy_input_ids,data_loader=unlabeled_dataloader,eval_callback=eval_callback)
    adaround_params = AdaroundParameters(data_loader=dataloader, num_batches=len(dataloader), default_num_iterations=32)
    auto_quant.set_adaround_params(adaround_params)
    model, accuracy = auto_quant.run_inference()
    sim, initial_accuracy = auto_quant.run_inference()
    model, optimized_accuracy, encoding_path = auto_quant.optimize(allowed_accuracy_drop=0.01)
    model, accuracy = auto_quant.run_inference()
    print(f"- Quantized Accuracy (before optimization): {initial_accuracy:.4f}")
    print(f"- Quantized Accuracy (after optimization):  {optimized_accuracy:.4f}")
