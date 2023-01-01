import os
import pickle
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from evaluate import load
from utils import compress, find_lastest_checkpoint
from bert_module import BertModule
from bart_module import BartModule
from numpy import mean


def predict(config, bert_model, bart_model, dataset, device):

    with torch.no_grad():
        for data_buf in tqdm(dataset):
            compressed_buf = compress(bert_model.compresser, data_buf, config.times, device, config.batch_size_inference)
            temp_inputs = torch.zeros(2, 1024, dtype=torch.long, device=device)
            input = [t.unsqueeze(0) for t in compressed_buf.export(temp_inputs, device=device)]
            output = bart_model.summarizer.generate(input_ids=input[0], attention_mask=input[1], num_beams=5, min_length=100, max_length=500, length_penalty=3, no_repeat_ngram_size=3)
            yield output[0], data_buf.summary['input_ids'][0]


def evaluate(config, mode):
    bertscore = load("bertscore")
    rouge = load('rouge')

    device = f'cuda:{config.gpus}'
    bert_model = BertModule.load_from_checkpoint(find_lastest_checkpoint(
        os.path.join(config.save_dir, 'bert_model', f'version_{config.version}', 'checkpoints')), ).to(device).eval()
    bart_model = BartModule.load_from_checkpoint(find_lastest_checkpoint(
        os.path.join(config.save_dir, 'bart_model', f'version_{config.version}', 'checkpoints'))).to(device).eval()
    if mode == 'test':
        with open(config.test_source, 'rb') as fin:
            dataset = pickle.load(fin)
    elif mode == 'validation':
        with open(config.validation_source, 'rb') as fin:
            dataset = pickle.load(fin)

    predictions, references = [], []

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    for prediction, reference in predict(config, bert_model, bart_model, dataset, device):
        predictions.append(prediction)
        references.append(reference)
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    references = tokenizer.batch_decode(references, skip_special_tokens=True)

    bertscore_result = bertscore.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")
    rouge_result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    result = {key: value * 100 for key, value in rouge_result.items()}
    result['bert-score-precision'] = mean(bertscore_result['precision'])
    result['bert-score-recall'] = mean(bertscore_result['recall'])
    result['bert-score-f1'] = mean(bertscore_result['f1'])
    print(result)