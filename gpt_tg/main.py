import torch
import random
import json
from transformers import (
    BertTokenizer, 
    GPT2LMHeadModel,
    GPT2Config,
    TextGenerationPipeline
    )
from torch.utils.data import (
    Dataset,
    DataLoader, 
    TensorDataset
)
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
from ipdb import set_trace

class TDDataset(Dataset):
    """
    GPT model dataset
    """

    def __init__(self, input_list, tokenizer, max_len):
        self.input_list = input_list
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):
        input_ids = self.input_list[index]
        if len(input_ids) >= self.max_len:
            _input_ids = input_ids[:self.max_len]
            _attention_mask = [1] * len(_input_ids)
        else:
            padding_length = self.max_len - len(input_ids)
            attention_mask = [1] * len(input_ids)
            _input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            _attention_mask = attention_mask + [0] * padding_length

        _input_ids = torch.tensor(_input_ids, dtype=torch.long)
        _attention_mask = torch.tensor(_attention_mask, dtype=torch.long)

        return _input_ids, _attention_mask

def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = rnn_utils.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=-100)

    return input_ids, attention_masks, labels

def preprocess_conversation(data, tokenizer):
    # tokenizer = BertTokenizer.from_pretrained(model_path)
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    dialogue_list = []
    for conver in data:
        input_ids = [cls_id] 
        for i in conver['conversation']:
            input_ids += tokenizer.encode(i['utterance'], add_special_tokens=False)
            input_ids.append(sep_id)
        dialogue_list.append(input_ids)
    
    return dialogue_list

def train_model(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # 微调模型
    model.to(device)
    model.train()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            # set_trace()
            inputs = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model.forward(inputs, attention_mask=attention_mask, labels=labels, return_dict=True)
            loss = outputs.loss
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()

        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, average loss: {average_loss:.2f}')
    model.save_pretrained(fintuned_path)

def predict():
    model = GPT2LMHeadModel.from_pretrained(fintuned_path)
    model.eval()
    input_text = "最近天气好好，想出去拍照片"
    num_generate = 5
    max_length = 50
    res = []

    for _ in range(num_generate):
        temperature = 0.1 * random.randint(1,9)
        input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt')
        output = model.generate(
                                input_ids=input_ids, 
                                max_length=max_length, 
                                pad_token_id=model.config.pad_token_id,
                                num_return_sequences=1,
                                temperature=temperature,
                                do_sample=True
                                )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        res.append(generated_text)
    for t in res:
        print("Generated Text:", t)


if __name__ == '__main__':
    pretrained_path = '/mnt/g/pretrained/gpt2-distil-chinese-cluecorpussmall'
    fintuned_path = '/mnt/g/my_finetuned/text_generate'
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    TRAIN = False
    if TRAIN:
        lr = 0.0001
        epochs = 3
        batch_size = 16
        MAX_LEN = 256
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GPT2LMHeadModel.from_pretrained(pretrained_path)
        data = []
        with open('./data/train.txt','r',encoding='utf-8') as f:
            for i in f.readlines():
                line = json.loads(i)
                data.append(line)

        dialogue_list = preprocess_conversation(data, tokenizer)
        train_dataset = TDDataset(dialogue_list, tokenizer, MAX_LEN)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=4,
            collate_fn=collate_fn,
            drop_last=True
        )
        train_model(model, train_loader)
    predict()

