import sys
import gc

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MinMaxScaler

import scipy.sparse as sp
import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import word_tokenize

import spacy
nlp = spacy.load("/t.gafurov/input/en-core-web-md/en_core_web_md/en_core_web_md-3.4.1")
from textblob import TextBlob

import warnings
warnings.filterwarnings('ignore')

def extract_features(text):
    ### TTR
    text = text.lower() 
    tokens = word_tokenize(text)
    tokens_count = len(tokens)
    type_count = len(set(tokens))
    ttr = type_count / tokens_count
    
    ### Flesch-Kincaid
#     fk = textstat.flesch_reading_ease(text)
    
    ### Polarity and Subjectivity
#     blob = TextBlob(text).sentiment
#     polarity = blob.polarity
#     subjectivity =  blob.subjectivity    
    
    ### Average sentence length
#     sentences = sent_tokenize(text)
#     avg_sen_len = 0
#     if len(sentences) != 0:
#         avg_sen_len = sum(len(word_tokenize(sentence)) for sentence in sentences) / len(sentences)
    
    return pd.Series({
        'ttr': ttr,
#         'fk': fk,
#         'polarity': polarity,
#         'subjectivity': subjectivity,
#         'avg_sen_len': avg_sen_len
    })

LOWERCASE = False
VOCAB_SIZE = 30522

raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

dataset = Dataset.from_pandas(df_test[['text']])

def train_corp_iter():
    for i in range(0, len(dataset), 1000):
        yield dataset[i: i + 1000]["text"]

raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
tokenized_texts_test = []

for text in tqdm(df_test['text'].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))

tokenized_texts_train = []

for text in tqdm(df_train['text'].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))

def dummy(text):
    return text

def dummy(text):
    return text
vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
    tokenizer = dummy,
    preprocessor = dummy,
    token_pattern = None, strip_accents='unicode')

vectorizer.fit(tokenized_texts_test)

# Getting vocab
vocab = vectorizer.vocabulary_

print(vocab)

vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                            analyzer = 'word',
                            tokenizer = dummy,
                            preprocessor = dummy,
                            token_pattern = None, strip_accents='unicode'
                            )

tf_train = vectorizer.fit_transform(tokenized_texts_train)
tf_test = vectorizer.transform(tokenized_texts_test)

del vectorizer
gc.collect()

for feature in df_train.columns:        
    if df_train[feature].min() < 0 or df_test[feature].min() < 0:
        scaler = MinMaxScaler()
        df_train[feature] = scaler.fit_transform(df_train[feature].values.reshape(-1, 1))
        df_test[feature] = scaler.transform(df_test[feature].values.reshape(-1, 1))

tf_train = sp.hstack((tf_train, sp.csr_matrix(df_train)))
tf_test = sp.hstack((tf_test, sp.csr_matrix(df_test)))

del df_train
del df_test
gc.collect()

if tf_test.shape[0] <= 5:
    sub.to_csv('submission.csv', index=False)
else:
    clf = MultinomialNB(alpha=0.1)
    sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber") 
    p6={'n_iter': 2500,
        'verbose': -1,
        'objective': 'cross_entropy',
        'metric': 'auc',
        'learning_rate': 0.01, 
        'colsample_bytree': 0.78,
        'colsample_bynode': 0.8, 
        'lambda_l1': 4.562963348932286, 
        'lambda_l2': 2.97485, 
        'min_data_in_leaf': 115, 
        'max_depth': 23, 
        'max_bin': 898}
    
    lgb=LGBMClassifier(**p6)
    cat=CatBoostClassifier(iterations=2000,
                           verbose=0,
                           l2_leaf_reg=6.6591278779517808,
                           learning_rate=0.1,
                           subsample = 0.4,
                           allow_const_label=True,loss_function = 'CrossEntropy')
    weights = [0.068,0.311,0.31,0.311]
 
    ensemble = VotingClassifier(estimators=[('mnb',clf),
                                            ('sgd', sgd_model),
                                            ('lgb',lgb), 
                                            ('cat', cat)
                                           ],
                                weights=weights, voting='soft', n_jobs=-1)
    ensemble.fit(tf_train, y_train)
    gc.collect()
    final_preds = ensemble.predict_proba(tf_test)[:,1]
    sub['generated'] = final_preds
    sub.to_csv('submission.csv', index=False)
    sub

