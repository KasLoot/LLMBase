import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import wandb
import os
from datetime import datetime

from dataclasses import dataclass, field
from nets.MiniChat.model import ModelConfig, MiniChat
from utils.dataset import LanguageModel_SFT_Dataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
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
    device: torch.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class TrainConfig:
    batch_size: int = 24
    epochs: int = 3
    learning_rate: float = 1e-4
    save_dir: str = "./checkpoints/"
    data_dir: str = "/workspace/datasets/pretrain_data_tokens.pt"
    model_config: ModelConfig = field(default_factory=ModelConfig)




class Trainer():
    def __init__(
            self,
            model: nn.Module,
            train_config: TrainConfig,
            train_dataloader: DataLoader,
            sampler: DistributedSampler,
            optimizer: torch.optim.Optimizer,
            scheduler,
            gpu_id: int,
            world_size: int
            ):
        self.gpu_id = gpu_id
        self.world_size = world_size
        self.model = model.to(gpu_id).to(torch.bfloat16)
        self.train_config = train_config
        self.train_dataloader = train_dataloader
        self.sampler = sampler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = DDP(model, device_ids=[gpu_id])
        self.loss_function = nn.CrossEntropyLoss(ignore_index=200000)

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


    def _run_batch(self, sep_index, input: torch.Tensor, targets: torch.Tensor):
        self.optimizer.zero_grad()
        logits: torch.Tensor = self.model(input)
        logits = logits.contiguous().view(-1, self.train_config.model_config.vocab_size)[:, sep_index+1:]
        targets = targets.view(-1).long()
        loss = self.loss_function(logits, targets)
        loss.backward()
        self.optimizer.step()
        
        total_loss = loss.clone().detach()
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        avg_loss = total_loss / self.world_size

        if self.gpu_id == 0:
            wandb.log({"loss": avg_loss.item(), "lr": self.scheduler.get_last_lr()[0]})

    def _run_epoch(self, epoch: int):
        self.train_dataloader.sampler.set_epoch(epoch)
        self.model.train()
        for batch in tqdm(self.train_dataloader, desc="Training"):
            sep_index, input, targets = batch
            input = input.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(sep_index, input, targets)

    def train(self):
        for epoch in tqdm(range(self.train_config.epochs), desc="Epochs"):
            self._run_epoch(epoch)
            self.scheduler.step()
            if self.gpu_id == 0:
                self.save_model(epoch)


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1250"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def load_train_objs(train_config: TrainConfig, rank: int):
    """
    Loads the training objects. To avoid issues with multiple processes
    reading the same file simultaneously, it's safe for each rank to
    instantiate its own dataset and model. Avoid broadcasting large
    Python objects (Dataset/Model/Optimizer) which can hang or be very
    slow when pickled across ranks.
    """
    # Each rank creates its own instances locally.
    print("initializing train_dataset")
    train_dataset = LanguageModel_SFT_Dataset(train_config.data_dir, train_config.model_config.sq_l)
    print("initializing model")
    model = MiniChat(train_config.model_config)
    print("initializing optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    print("initializing scheduler")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config.epochs, eta_min=0)

    return train_dataset, model, optimizer, scheduler


def main(rank: int, world_size: int):
    train_config = TrainConfig()
    ddp_setup(rank, world_size)
    if rank == 0:
        wandb.init(
            project="LLMBase-DDP-Pretrain-MiniChat",
            config={
                "learning_rate": train_config.learning_rate,
                "epochs": train_config.epochs,
                "batch_size": train_config.batch_size,
                "model_config": train_config.model_config.__dict__,
            }
        )
    print(f"[{rank}] Loading objects")
    dataset, model, optimizer, scheduler = load_train_objs(train_config, rank)
    print(f"[{rank}] Loading DistributedSampler")
    sampler = DistributedSampler(dataset)
    print(f"[{rank}] Initializing DataLoader")
    data_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=sampler
    )
    print(f"[{rank}] Initializing Trainer")
    trainer = Trainer(
        model=model,
        train_dataloader=data_loader,
        sampler=sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        gpu_id=rank,
        train_config=train_config,
        world_size=world_size
    )
    print(f"[{rank}] Start training")
    trainer.train()
    if rank == 0:
        wandb.finish()
    destroy_process_group()


if __name__ == "__main__":
    wandb.login(key="9c5a4f0ff5bbd5c1ac287e84272b09e4dd0e023a")
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
