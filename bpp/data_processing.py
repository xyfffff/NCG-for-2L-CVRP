import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn.utils.rnn import pad_sequence
import logging

def load_batch_data(batch_path):
    # Load labels from CSV
    labels = {}
    with open(os.path.join(batch_path, 'results.csv'), 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            labels[row[0]] = int(row[1])

    # Process .tr files
    instances = []
    file_names = []
    for file in os.listdir(os.path.join(batch_path, 'instances')):
        if file in labels and labels[file] != 3:
            with open(os.path.join(batch_path, 'instances', file), 'r') as f:
                lines = f.readlines()
                H, W = map(float, lines[1].strip().split())
                data = [torch.tensor((float(line.split()[1])/H, float(line.split()[2])/W, float(line.split()[3])), dtype=torch.float32) for line in lines[2:]]
                data_stacked = torch.stack(data)
                instances.append((data_stacked, torch.tensor([labels[file]], dtype=torch.float32)))
                file_names.append(file)

    return instances, file_names


# Custom Dataset
class BinPackingDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

def collate_fn(batch):
    data, labels = zip(*batch)
    data = pad_sequence(data, batch_first=True, padding_value=-1)
    labels = torch.stack(labels)
    return data, labels

def load_data(data_list, train_batch_size, val_batch_size=512, test_batch_size=512, filter_string=None):
    instances, file_names = [], []
    for batch in data_list:
        batch_instances, batch_file_names = load_batch_data('../' + batch)
        instances.extend(batch_instances)
        file_names.extend(batch_file_names)

    # Compute class weights
    positive_samples = sum([label for _, label in instances])
    negative_samples = len(instances) - positive_samples
    total_samples = len(instances)

    weight_for_0 = total_samples / (2 * negative_samples)
    weight_for_1 = total_samples / (2 * positive_samples)
    pos_weight = torch.tensor([weight_for_1 / weight_for_0])
    # pos_weight = torch.tensor([1])

    dataset = BinPackingDataset(instances)
    logging.info("Positive cnt: {}, Negative cnt: {}".format(positive_samples.item(), negative_samples.item()))
    logging.info("Positive ratio: {}, Negative ratio: {}".format((positive_samples/total_samples).item(), (negative_samples/total_samples).item()))

    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    print("Length of test set: ", len(test_dataset))

    if filter_string:
        test_indices = test_dataset.indices
        filtered_test_indices = [idx for idx in test_indices if filter_string in file_names[idx]]
        test_dataset = Subset(dataset, filtered_test_indices)
        print("( Filtered ) Length of test set: ", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, pos_weight