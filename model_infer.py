import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch.nn as nn
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "philschmid/tiny-bert-sst2-distilled"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)


# summary(model,input_data=[duumy_input_ids,duumy_attention_mask])

# 将模型移动到 GPU 并设置为评估模式
model.to(device)
model.eval()

batch = 1000

print(isinstance(model, nn.Module))

# 加载数据集
dataset = load_dataset('glue', 'sst2')['train']

for i in range(batch):
    print(f'batch_{i}:')
    # 示例：对数据集进行分词
    inputs = tokenizer(dataset[i]['sentence'], return_tensors='pt', padding=True, truncation=True, max_length=128)
    print(f'sentence: ', dataset[i]['sentence'])

    # 打印输入参数
    print("Input IDs:", inputs['input_ids'])
    print("Attention Mask:", inputs['attention_mask'])

    # 将输入数据移动到 GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 禁用梯度计算
    with torch.no_grad():
        # 模型推理
        outputs = model(**inputs)
        logits = outputs.logits

    # 打印 logits
    print("Logits:", logits)

    # 转化为标签
    predictions = torch.argmax(logits, dim=-1)
    print("Predicted Label:", predictions.item())

