###########################################
################ importing ################
###########################################

import os
import gc
import random
import warnings
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1980)

from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig


warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = f"/t.gafurov/input/llm-detect-ai-generated-text/"
EXP = "exp20"

model_paths = [
'/tinkoff/input/llm-detect-deberta-v3-large/pytorch/5folds/1/microsoft-deberta-v3-large_fold0_best.pth',
'/tinkoff/input/llm-detect-deberta-v3-large/pytorch/5folds/1/microsoft-deberta-v3-large_fold1_best.pth',
'/tinkoff/input/llm-detect-deberta-v3-large/pytorch/5folds/1/microsoft-deberta-v3-large_fold2_best.pth',
'/tinkoff/input/llm-detect-deberta-v3-large/pytorch/5folds/1/microsoft-deberta-v3-large_fold3_best.pth',
'/tinkoff/input/llm-detect-deberta-v3-large/pytorch/5folds/1/microsoft-deberta-v3-large_fold4_best.pth',
]

print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
print(f"torch.__version__: {torch.__version__}")
print(f"torch cuda version: {torch.version.cuda}")
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoTokenizer, AutoConfig


#########################################
################ Classes ################
#########################################

class Config:
    debug = False
    num_workers = 4
    llm_backbone =  "/kaggle/input/huggingfacedebertav3variants/deberta-v3-large"
    tokenizer_path = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=True, trust_remote_code=True,
    )
    batch_size = 8
    max_len = 512
    seed = 42
    num_labels = 1
    gradient_checkpointing = False
CFG = Config()

test = pd.read_csv(f"{data_dir}test_text.csv")
# test = pd.read_csv(f"{data_dir}train_text.csv")

test['full_text'] = test['text']

test["tokenize_length"] = [
    len(CFG.tokenizer(text)["input_ids"]) for text in test["full_text"].values
]
test = test.sort_values("tokenize_length", ascending=False).reset_index(drop=True)
test.drop(["tokenize_length"], axis=1, inplace=True)

def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=CFG.max_len,
        pad_to_max_length=True,
        truncation="longest_first",
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df["full_text"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs['token_type_ids'],
        }

class CollateCls:
    def __init__(self, cfg):
        self.tokenizer = cfg.tokenizer
        self.cfg = cfg
        
    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output["token_type_ids"] = [sample["token_type_ids"] for sample in batch]
        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [
                list(s) + (batch_max - len(s)) * [self.tokenizer.pad_token_id]
                for s in output["input_ids"]
            ]
            output["attention_mask"] = [
                list(s) + (batch_max - len(s)) * [0] for s in output["attention_mask"]
            ]
            output["token_type_ids"] = [list(s) + (batch_max - len(s)) * [0] for s in output["token_type_ids"]]

        else:
            output["input_ids"] = [
                (batch_max - len(s)) * [self.tokenizer.pad_token_id] + list(s)
                for s in output["input_ids"]
            ]
            output["attention_mask"] = [
                (batch_max - len(s)) * [0] + list(s) for s in output["attention_mask"]
            ]
            output["token_type_ids"] = [(batch_max - len(s)) * [0] + list(s) for s in output["token_type_ids"]]
        
        
        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        output["token_type_ids"] = torch.tensor(output["token_type_ids"], dtype=torch.long)
        
        return output


#######################################
################ Model ################
#######################################

test_dataset = TestDataset(CFG, test)

test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False, 
    collate_fn=CollateCls(CFG),
    num_workers=0, #CFG.num_workers,
    pin_memory=False,
    drop_last=False,
)

config_path =  "/t,gafurov/input/llm-detect-deberta-v3-large/pytorch/5folds/1/config.pth"

config = torch.load(config_path)
config

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            self.config.add_pooling_layer = False 
            # LOGGER.info(self.config)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        
        self.model.resize_token_embeddings(len(CFG.tokenizer))
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.num_labels)
        self._init_weights(self.fc)

def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, input_ids, attention_mask, token_type_ids ):
        outputs = self.model( input_ids, attention_mask, token_type_ids)
        last_hidden_states = outputs[0] 
        feature = last_hidden_states[:, 0, :] ## CLS token
        return feature

    def forward(self,  input_ids, attention_mask, token_type_ids):
        feature = self.feature( input_ids, attention_mask, token_type_ids)
        output = self.fc(feature)
        return output.squeeze(-1)

###########################################
################ inference ################
###########################################

def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(
                enabled=True, dtype=torch.float16, cache_enabled=True
            ):
                y_preds = model(inputs["input_ids"], inputs["attention_mask"],inputs['token_type_ids']  )
        preds.append(y_preds.to("cpu").numpy().astype(np.float32))
    predictions = np.concatenate(preds)
    return predictions

predictions = []
for i in range(len(model_paths)):
    model = CustomModel(CFG, config_path=config_path, pretrained=False)
    state = torch.load(model_paths[i], map_location=torch.device("cpu"))
    model.load_state_dict(state["model"], strict=False)
    prediction = inference_fn(test_loader, model, device)
    predictions.append(prediction)
    del model, state, prediction
    gc.collect()
    torch.cuda.empty_cache()
predictions = np.mean(predictions, axis=0)
predictions = torch.sigmoid(torch.tensor(predictions)).numpy()

