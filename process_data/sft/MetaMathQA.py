import torch
import json

import os

DATA_PATH = "/Users/liuyuxin/Documents/datasets/meta-math/MetaMathQA/MetaMathQA-395K.json"

with open(DATA_PATH) as MetaMathQA_data:
    MetaMathQA_data = json.load(MetaMathQA_data)[0]






MetaMathQA_data.close()