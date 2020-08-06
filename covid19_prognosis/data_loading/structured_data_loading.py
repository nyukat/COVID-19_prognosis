import torch
from torch.utils.data import Dataset

class RandomDataLoader(Dataset):
    """
    Class that generates a random set of structured data that represents the clinical variables data
    """
    def __init__(self, parameters):
        self.labels = [torch.randint(low=0, high=2, size=(parameters["number_classes"],)) for _ in range(parameters["data_num"])]
        self.features = [torch.rand(size=(parameters["number_features"],)).tolist() for _ in range(parameters["data_num"])]

    def __getitem__(self, index):
        row = self.features[index]
        label = self.labels[index]
        return index, row, label

    def __len__(self):
        return len(self.labels)
