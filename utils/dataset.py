import torch
from torch.utils.data import Dataset



class LanguageModel_SFT_Dataset(Dataset):
    """Custom PyTorch Dataset for next-token prediction."""
    def __init__(self, path):
        """
        Args:
            tokenized_text (torch.Tensor): A tensor of tokenized text
                                           of shape (num_samples, sequence_length).
        """
        self.samples = torch.load(path)


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index and splits it into
        input and target tensors.
        """
        # Get the full sequence for the given index
        sequence = self.samples[idx]

        # The input is the sequence up to the last token
        input_ids = sequence[:-1]

        # The target is the sequence from the second token onwards
        target_ids = sequence[1:]

        return input_ids, target_ids
    

class LanguageModelDataset(Dataset):
    """Custom PyTorch Dataset for next-token prediction."""
    def __init__(self, path):
        """
        Args:
            tokenized_text (torch.Tensor): A tensor of tokenized text
                                            of shape (num_samples, sequence_length).
        """
        self.samples = torch.load(path)


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index and splits it into
        input and target tensors.
        """
        # Get the full sequence for the given index
        sequence = self.samples[idx]

        # The input is the sequence up to the last token
        input_ids = sequence[:-1]

        # The target is the sequence from the second token onwards
        target_ids = sequence[1:]

        return input_ids, target_ids