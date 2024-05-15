import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from collections import Counter
from itertools import chain

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from transformers import (
    PreTrainedTokenizerFast,
    Trainer, 
    TrainingArguments,
    DistilBertForSequenceClassification, 
    DistilBertConfig
)
from datasets import Dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from scipy.sparse import vstack as spvstack
from scipy.special import softmax

class BPETokenizer:
    ST = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tok.normalizer = normalizers.Sequence([normalizers.NFC()])
        self.tok.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.tok.post_processor = processors.TemplateProcessing(
            single="[CLS] $A",
            special_tokens=[("[CLS]", 1)],
        )
        
    @classmethod
    def chunk_dataset(cls, dataset, chunk_size=1_000):
        for i in range(0, len(dataset), chunk_size):
            yield dataset[i : i + chunk_size]["text"]
        
    def train(self, data):
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.ST)
        dataset = Dataset.from_pandas(data[["text"]])
        self.tok.train_from_iterator(self.chunk_dataset(dataset), trainer=trainer)
        return self
    
    def tokenize(self, data):
        tokenized_texts = []
        for text in tqdm(data['text'].tolist()):
            tokenized_texts.append(self.tok.encode(text))
        return tokenized_texts
    
    def get_fast_tokenizer(self, max_length):
        return PreTrainedTokenizerFast(
            tokenizer_object=self.tok,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            model_max_length=max_length
        )

class DAIGTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, labels = None):
        self.tokenized_data = tokenized_data
        if labels is not None:
            self.labels = labels
        else:
            self.labels = np.zeros(self.tokenized_data.input_ids.shape[0], dtype="int")
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_data.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return self.labels.shape[0]
    
    @classmethod
    def create_tokenized_dataset(cls, tknzr, df):
        tokenized_data = tknzr(
            df.text.tolist(), 
            max_length=tknzr.model_max_length, 
            padding="max_length", 
            return_tensors="pt",
            truncation=True
        )
        if "label" in df:
            labels = df.label.values
        else:
            labels = None
        return cls(tokenized_data, labels=labels)
    
    
def compute_roc_auc(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if labels.std() < 1E-8: # only one class present in dataset
        return {"roc_auc": 0.0}
    ps = softmax(logits, axis=-1)[:,1]
    return {"roc_auc": roc_auc_score(labels, ps)}

%%time
bpe_tok = BPETokenizer(10_000).train(pd.concat((train, test)).reset_index(drop=True))

tokenized_texts_train = [x.tokens for x in bpe_tok.tokenize(train)]
tokenized_texts_test = [x.tokens for x in bpe_tok.tokenize(test)]

def noop(text):
    return text

print("Fitting vectorizer")
vectorizer = TfidfVectorizer(
    ngram_range=(3, 5), 
    lowercase=False, 
    use_idf=False,
    # sublinear_tf=True,
    analyzer='word',
    tokenizer=noop,
    preprocessor=noop,
    token_pattern=None, 
    vocabulary=None,
    strip_accents='unicode'
)
vectorizer.fit(tokenized_texts_test)

x_tr = vectorizer.transform(tokenized_texts_train)
x_te = vectorizer.transform(tokenized_texts_test)
y_tr = train.label.values

print("Training classifier")
clf = SGDClassifier(
    loss="modified_huber",
    max_iter=12_500,
    tol=1E-4,
).fit(x_tr, y_tr)

seq_length = 1024
tokenizer = bpe_tok.get_fast_tokenizer(seq_length)
db_config = DistilBertConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=seq_length,
    n_layers=3,
    n_heads=4,
    pad_token_id=tokenizer.pad_token_id
)

%%time
tr_df, val_df = train_test_split(train, random_state=113, test_size=0.2)

sl = slice(None) # DEBUGGING: slice(16, 32)

train_dataset = DAIGTDataset.create_tokenized_dataset(tokenizer, tr_df[sl])
val_dataset = DAIGTDataset.create_tokenized_dataset(tokenizer, val_df[sl])

print(f"train size = {len(train_dataset)}, validation size = {len(val_dataset)}")

db_model = DistilBertForSequenceClassification(db_config)
training_args = TrainingArguments(
    output_dir="results",            # output directory
    num_train_epochs=1,              # total number of training epochs
    # max_steps=11,
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir="logs",              # directory for storing logs
    logging_steps=100,
    report_to="none",
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    greater_is_better=True,
)
trainer = Trainer(
    model=db_model,                       
    args=training_args,                   # training arguments, defined above
    train_dataset=train_dataset,          # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=compute_roc_auc,
)

trainer.train()

%%time
test_dataset = DAIGTDataset.create_tokenized_dataset(tokenizer, test)
test_predictions = trainer.predict(test_dataset)
p_db = softmax(test_predictions.predictions, axis=-1)[:, 1]
p_db
