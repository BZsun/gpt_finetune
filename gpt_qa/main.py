import torch
import random
import json
from transformers import (
    GPT2Tokenizer,
    BertTokenizer, 
    GPT2ForQuestionAnswering,
    GPT2Config
)
from torch.utils.data import (
    Dataset,
    DataLoader, 
    TensorDataset
)
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
from ipdb import set_trace

class QADataset(Dataset):
    def __init__(self, contexts, questions, answers, tokenizer, max_len):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        question = self.questions[idx]
        answer = self.answers[idx]
        inputs = self.tokenizer.encode_plus(question, context, add_special_tokens=True, \
            padding='max_length', pad_to_max_length=True, max_length=self.max_len, \
            truncation=True, return_tensors='pt')
        # start_positions = torch.tensor([inputs['input_ids'].tolist()[0].index(self.tokenizer.bos_token_id)])
        # end_positions = torch.tensor([inputs['input_ids'].tolist()[0].index(self.tokenizer.eos_token_id)])
        if self.tokenizer.bos_token_id in inputs['input_ids'].tolist()[0] :
            start_positions = torch.tensor([inputs['input_ids'].tolist()[0].index(self.tokenizer.bos_token_id)])
        else:
            start_positions = torch.tensor([-1])
        if self.tokenizer.eos_token_id in inputs['input_ids'].tolist()[0]:
            end_positions = torch.tensor([inputs['input_ids'].tolist()[0].index(self.tokenizer.eos_token_id)])
        else:
            end_positions = torch.tensor([-1])
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'start_positions': start_positions.flatten(),
            'end_positions': end_positions.flatten(),
        }

def split_text_into_chunks(text, max_length):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return chunks

def preprocess_data(jsonfile, max_length=128):
    with open(jsonfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    answers = []
    questions = []
    contexts = []
    length = max_length // 2

    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            # context_chunks = split_text_into_chunks(context, max_length)
            if len(context.split()) <= 200:
                for qa in paragraph['qas']:
                    question = qa['question']
                    if qa['answers']:
                        answer = qa['answers'][0]['text']
                    elif qa['plausible_answers']:
                        answer = qa['plausible_answers'][0]['text']
                    
                    if question and answer:
                        # question_chunks = [question] * len(context_chunks)
                        # answer_chunks = [answer] * len(context_chunks)
                        # contexts.extend(context_chunks)
                        # questions.extend(question_chunks)
                        # answers.extend(answer_chunks)
                        
                        _start = context.find(answer)
                        _end = len(answer) + _start
                        start = _start - length if _start - length > 0 else 0
                        end = _end + length if _end + length < len(context) - 1 else len(context) - 1
                        _context = context[start: end]
                        contexts.append(_context)
                        questions.append(question)
                        answers.append(answer)
    return contexts, questions, answers

def train_model(model, dataloader):
    model.train()
    epochs = 3
    lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print('Epoch:', epoch+1, 'Loss:', average_loss)

    model.save_pretrained(fintuned_path)
    tokenizer.save_pretrained(fintuned_path)

def predict():
    tokenizer = GPT2Tokenizer.from_pretrained(fintuned_path)
    model = GPT2ForQuestionAnswering.from_pretrained(fintuned_path)
    model.to(device)
    model.eval()
    pad_token_id = tokenizer.eos_token_id
    context = 'the number of sequence is 5.'
    question = 'what is the sequence number?'

    input_ids = tokenizer.encode(question, context, add_special_tokens=True, truncation=True, max_length=256, padding=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    
    # 使用模型进行预测
    with torch.no_grad():
        output = model(input_ids=input_ids)
        start_logits = output['start_logits']
        end_logits = output['end_logits']

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits)
    answer = ' '.join(all_tokens[answer_start: answer_end + 1])
    answer = tokenizer.convert_tokens_to_string(answer)

    print('context: ', context)
    print('question:', question)
    print("predict answer:", answer)

if __name__ == '__main__':
    pretrained_path = '/mnt/g/pretrained/gpt2'
    fintuned_path = '/mnt/g/my_finetuned/qa'
    train_file = './data/train-v2.0.json'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TRAIN = False
    if TRAIN:
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_path)
        # pad_token_id = tokenizer.pad_token_id
        tokenizer.pad_token = tokenizer.eos_token
        contexts, questions, answers = preprocess_data(train_file)
        dataset = QADataset(contexts, questions, answers, tokenizer, max_len=256)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        model = GPT2ForQuestionAnswering.from_pretrained(pretrained_path)
        model.resize_token_embeddings(len(dataset.tokenizer)) # 调整模型的嵌入层以适应新的标记
        model.to(device)
        train_model(model, dataloader)
    else:
        predict()
