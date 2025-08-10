import torch
from torch.utils.data import Dataset, DataLoader

from nets.demo_model.model import MiniChat

from utils.tokenizer import get_tokenizer

from nets.demo_model.model import ModelConfig

from utils.dataset import process_data, LanguageModelDataset

from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"








dummy_train_text = [["Hello, this is Steven, how are you?"],
                    ["This is an apple."],
                    ["There are 3 chilren achieved 1003842 score on the leaderboard"]]



def train():

    CFG = ModelConfig()

    # tokenizer = get_tokenizer()
    # print("tokenizer initialized")
    # process_data(tokenizer, CFG)


    dataset = LanguageModelDataset(path="/home/yuxin/datasets/LLMBase/train_tiny_textbook.pt")

    train_data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)


    # print(f"input shape: {inputs.shape}, contigous: {inputs.is_contiguous()}\n{inputs[0]}")
    # print(f"targets shape: {targets.shape}, contigous: {targets.is_contiguous()}\n{targets[0]}")

    

    model = MiniChat(CFG=CFG).to(device=device)

    optmizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=200000)


    for epoch in tqdm(range(20)):
        for batch in tqdm(train_data_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            optmizer.zero_grad()
            logits: torch.Tensor = model(inputs)  # Raw logits from model
            print(logits.shape)
            logits = logits.contiguous().view(-1, CFG.vocab_size)
                

            loss = loss_func(logits, targets.long())
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
            loss.backward()
            optmizer.step()

    



train()





# model = DemoTextModel(CFG).to(device).to(torch.bfloat16)
# x = torch.randint(0, 100000, (CFG["batch_size"], CFG["sq_l"])).to(device)


# mem_p = torch.cuda.memory_allocated()
# y = model(x)
# peak_mem = torch.cuda.max_memory_allocated()
# print(f"memory usage: {(peak_mem - mem_p)/1e6} Mb")