import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import wandb
import os
from datetime import datetime

from dataclasses import dataclass
from nets.MiniChat.model import ModelConfig, MiniChat
from utils.dataset import LanguageModelDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm

@dataclass
class ModelConfig:
    sq_l: int = 2048
    d_model: int = 2048
    num_h: int = 16
    hidden_size: int = 4096
    drop_rate: float = 0.1
    vocab_size: int = 201088
    transformer_blocks: int = 6


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 1e-4
    save_dir: str = "./checkpoints/"
    data_dir: str = "/Users/liuyuxin/Documents/LLMBase/data_samples/demo_pretrain_data_tokens.pt"
    model_config: ModelConfig = ModelConfig()




class Trainer():
    def __init__(
            self,
            model: nn.Module,
            train_config: TrainConfig,
            train_dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int
            ):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id).to(torch.bfloat16)
        self.train_config = train_config
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.model = DDP(model, device_ids=[gpu_id])
        self.loss_function = nn.CrossEntropyLoss(ignore_index=200000)
        self.train_config = train_config

    def save_model(self, epoch: int):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.train_config.model_config.__dict__,
            "train_config": self.train_config.__dict__
        }
        save_path = os.path.join(self.train_config.save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")


    def _run_batch(self, input: torch.Tensor, targets: torch.Tensor):
        self.optimizer.zero_grad()
        logits: torch.Tensor = self.model(input)
        logits = logits.contiguous().view(-1, self.train_config.model_config.vocab_size)
        targets = targets.view(-1).long()
        loss = self.loss_function(logits, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self):
        self.model.train()
        for batch in tqdm(self.train_dataloader, desc="Training"):
            input, targets = batch
            input = input.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(input, targets)

    def train(self):
        for epoch in tqdm(range(self.train_config.epochs), desc="Epochs"):
            self._run_epoch()
            if self.gpu_id == 0:
                self.save_model(epoch)


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def load_train_objs(train_config: TrainConfig):
    train_dataset = LanguageModelDataset(train_config.data_dir, train_config.model_config.sq_l)
    model = MiniChat(train_config.model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    return train_dataset, model, optimizer


def main(rank: int, world_size: int):
    train_config = TrainConfig()
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs(train_config)
    data_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(dataset)
    )
    trainer = Trainer(
        model=model,
        model_config=train_config.model_config,
        train_dataloader=data_loader,
        optimizer=optimizer,
        gpu_id=rank,
        train_config=train_config
    )
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
