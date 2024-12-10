import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from aimet_torch.model_preparer import prepare_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel,save_checkpoint,load_checkpoint
from aimet_torch.onnx_utils import OnnxExportApiArgs
from collections import namedtuple
import torch.nn as nn 
import time
from transformers import AdamW

Output = namedtuple('Output', ['logits'])


class CustomModel(nn.Module):
    def __init__(self, model_name):
        super(CustomModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()
        # self.model = prepare_model(self.model)

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
        self.raw_model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()
        self.model = CustomModel(model_name).cuda()
        # _ = fold_all_batch_norms(self.model, input_shapes=[(1, 128),(1, 128)])

    def get_val_dataloader(self, sentences, labels, batch_size=8) -> DataLoader:
        dataset = TextDataset(sentences, labels, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return data_loader

    def evaluate_with_rawModel(self, sentences, labels, use_cuda: bool,dummy_inputs = None) -> float:
        model = AutoModelForSequenceClassification.from_pretrained('philschmid/tiny-bert-sst2-distilled').eval().cuda()
        val_loader = self.get_val_dataloader(sentences, labels)
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.cuda() for k, v in batch.items() if k != 'label'} if use_cuda else batch
                labels = batch['label'].cuda() if use_cuda else batch['label']
                if dummy_inputs != None:
                   inputs = {'input_ids':dummy_inputs[0].to('cuda'),'attention_mask':dummy_inputs[1].to('cuda')} 
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        accuracy = correct_predictions / total_predictions
        return accuracy

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
        self.model.cuda()
        sim = QuantizationSimModel(model=self.model,
                           quant_scheme=QuantScheme.training_range_learning_with_tf_enhanced_init,   # QAT with range learning
                           dummy_input=(duumy_input_ids,duumy_attention_mask),
                           default_output_bw=bit_width,
                           default_param_bw=bit_width)
        return sim


    def finetune(self,sim_model:QuantizationSimModel,train_sentences, train_labels, epochs=5, learning_rate=2e-5, batch_size=8):
        print('start qat')
        train_loader = self.get_val_dataloader(train_sentences, train_labels, batch_size=batch_size)
        optimizer = AdamW(sim_model.model.parameters(), lr=learning_rate)
        global exist_finetune
        sim_model.model.train()
        ckpt_file_path = '/root/projects/SNPE/aimetTest/aimet/checkpoints/best_qat.pth'
        best_avg_loss = 100000
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                logits = sim_model.model(**inputs).logits
                raw_logits = self.raw_model(**inputs).logits
                loss = torch.nn.functional.mse_loss(logits, raw_logits)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            if best_avg_loss > avg_loss:
                best_avg_loss = avg_loss
                save_checkpoint(sim_model, ckpt_file_path)

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')

            sim_model = load_checkpoint(ckpt_file_path)

          
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
epochs = 5
bit_width = 8
exist_finetune = False


if __name__ == "__main__":
    # dataset = load_dataset('glue', 'sst2')['train']
    dataset = load_dataset('glue', 'sst2')['validation']
    sentences = [example['sentence'] for example in dataset][:sample_size]
    labels = [example['label'] for example in dataset][:sample_size]
    use_cuda = torch.cuda.is_available()
    pipeline = ModelDataPipeline()
    pipeline.evaluate_with_rawModel(sentences,labels,use_cuda)
    iter = 1
    duumy_input_ids = torch.randint(0,1000, (1, 128)).to(torch.int64).to('cpu')
    duumy_attention_mask = torch.ones((1, 128)).to(torch.int64).to('cpu')
    dummy_inputs = (duumy_input_ids,duumy_attention_mask)
    sim_model = pipeline.quantize()
    sim_model.compute_encodings(forward_pass_callback=pass_calibration_data,forward_pass_callback_args=(pipeline,sentences,labels))
    # QAT Benchmark
    pipeline.finetune(sim_model,sentences,labels)
    # start_time = time.time()
    # accuracys = 0
    # for i in range(iter):
    #     accuracy = pipeline.evaluate(sentences, labels, use_cuda=use_cuda)
    #     accuracys += accuracy
    # end_time = time.time()
    # print(f'qat: infer time: {(end_time - start_time)/iter}  average_accuracy: {accuracys / iter}')
    sim_model.export(path='/root/projects/SNPE/aimetTest/models/sim_models',filename_prefix='tinyBert_qat',dummy_input=(duumy_input_ids,duumy_attention_mask),onnx_export_args=OnnxExportApiArgs(opset_version=14),use_embedded_encodings=True)
    # sim_model.export(path='/root/projects/SNPE/aimetTest/models/sim_models',filename_prefix='tinyBert_qat',dummy_input=(duumy_input_ids,duumy_attention_mask))
    print('finished!')

