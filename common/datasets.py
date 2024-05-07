from torch.utils.data import Dataset


class GenericDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.curr_idx = 0

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_idx < len(self.data):
            self.curr_idx += 1
            return self.data[self.curr_idx - 1], self.targets[self.curr_idx - 1]
        else:
            raise StopIteration
