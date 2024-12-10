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



Output = namedtuple('Output', ['logits'])


class CustomModel(nn.Module):
    def __init__(self, model_name):
        super(CustomModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).cuda()

    def forward(self, input_ids, attention_mask):
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
        self.model = CustomModel(model_name).cuda()
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
                           quant_scheme=QuantScheme.post_training_tf,   # CLE + Bias Correction
                           dummy_input=(duumy_input_ids,duumy_attention_mask),
                           default_output_bw=bitwidth,
                           default_param_bw=bitwidth)
        return sim


    def finetune(self,sim_model,train_sentences, train_labels, epochs=100, learning_rate=2e-5, batch_size=8):
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


# sample_size = 10000
sample_size = 100
bitwidth = 8

if __name__ == "__main__":
    # dataset = load_dataset('glue', 'sst2')['train']
    dataset = load_dataset('glue', 'sst2')['validation']
    # dataset = load_dataset('glue', 'sst2')['validation']
    sentences = [example['sentence'] for example in dataset][:sample_size]
    labels = [example['label'] for example in dataset][:sample_size]
    use_cuda = torch.cuda.is_available()
    pipeline = ModelDataPipeline()
    # PTQ Benchmark
    start_time = time.time()
    sim_accuracy = 0
    iter = 3
    
    dataloader = pipeline.get_val_dataloader(sentences=sentences,labels=labels)
    duumy_input_ids = torch.randint(0,1000, (1, 128)).to(torch.int64).to('cpu')
    duumy_attention_mask = torch.ones((1, 128)).to(torch.int64).to('cpu')
    dummy_inputs = (duumy_input_ids,duumy_attention_mask)
    
    dummy_input_shape = [(1,128),(1,128)]
    equalize_model(pipeline.model,input_shapes = dummy_input_shape , dummy_input=dummy_inputs)
    # bc_params = QuantParams(weight_bw=bitwidth, act_bw=bitwidth, round_mode="nearest",quant_scheme=QuantScheme.post_training_tf_enhanced)
    # correct_bias(pipeline.model, bc_params, num_quant_samples=sample_size,data_loader=dataloader, num_bias_correct_samples=sample_size)

    sim_model = pipeline.quantize()
    sim_model.compute_encodings(forward_pass_callback=pass_calibration_data,forward_pass_callback_args=(pipeline,sentences,labels))

    # PTQ Benchmark
    # start_time = time.time()
    # sim_accuracy = 0
    # for i in range(iter):
    #     accuracy = pipeline.evaluate_simModel(sentences, labels, use_cuda,sim_model)
    #     sim_accuracy += accuracy
    # end_time = time.time()
    # print(f'sim:  infer time: {(end_time - start_time) / iter}  average_accuracy: {sim_accuracy / iter}')

    # QAT Benchmark
    pipeline.finetune(sim_model,sentences,labels)
    start_time = time.time()
    accuracys = 0
    for i in range(iter):
        accuracy = pipeline.evaluate(sentences, labels, use_cuda=use_cuda)
        accuracys += accuracy
    end_time = time.time()
    print(f'qat: infer time: {(end_time - start_time)/iter}  average_accuracy: {accuracys / iter}')
        
    sim_model.export(path='/root/projects/SNPE/aimetTest/models/sim_models',filename_prefix='tinyBert_qat',dummy_input=(duumy_input_ids,duumy_attention_mask))
    print('finished!')
