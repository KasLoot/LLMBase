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
    def __init__(self, path, seq_length):
        """
        Args:
            tokenized_text (torch.Tensor): A tensor of tokenized text
                                            of shape (num_samples, sequence_length).
        """
        self.samples = torch.load(path)
        self.seq_length = seq_length
        self.samples = torch.tensor(list(self.split_into_chunks(self.samples))[:-1])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)
    
    def split_into_chunks(self, tokens):
        """
        Splits the samples into chunks of the specified sequence length.
        """
        for sequence in tokens:
            for i in range(0, len(sequence), self.seq_length+1):
                yield sequence[i:i+self.seq_length]

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index and splits it into
        input and target tensors.
        """
        # Get the full sequence for the given index
        sequence = self.samples[idx]

        input = sequence[:-1]
        target = sequence[1:]

        return input, target