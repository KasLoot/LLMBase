from utils.tokenizer import get_tokenizer
from nets.demo_model.model import ModelConfig
from utils.dataset import process_data


CFG = ModelConfig()

def pre_process_data():
    tokenizer = get_tokenizer()
    print("tokenizer initialized")
    process_data(tokenizer, CFG)











pre_process_data()