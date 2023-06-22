## Standard libraries
import os
import json
import math
import numpy as np
import time

## Imports for plotting
import matplotlib.pyplot as plt
import torch_geometric
from IPython.display import set_matplotlib_formats
from matplotlib.colors import to_rgb
import matplotlib

from src.Models.ezGNN import GNNClassifier

matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()
sns.set()

## Progress bar
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from Models.GraphNN import *
# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data/datasets/"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../models/"

# Setting the seed
pl.seed_everything(42)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)


def train_graph_classifier(model_name, tu_dataset, graph_train_loader, graph_val_loader, graph_test_loader, **model_kwargs):
    pl.seed_everything(42)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=500,
                         enable_progress_bar=False)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"GraphLevel{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = GraphLevelGNN(c_in=tu_dataset.num_node_features,
                              c_out=1 if tu_dataset.num_classes==2 else tu_dataset.num_classes,
                              **model_kwargs)
        trainer.fit(model, graph_train_loader, graph_val_loader)
        model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # Test best model on validation and test set
    train_result = trainer.test(model, graph_train_loader, verbose=False)
    test_result = trainer.test(model, graph_test_loader, verbose=False)
    result = {"test": test_result[0]['test_acc'], "train": train_result[0]['test_acc']}
    return model, result



def run():
    tu_dataset = torch_geometric.datasets.TUDataset(root=DATASET_PATH, name="MUTAG")

    print("Data object:", tu_dataset.data)
    print("Length:", len(tu_dataset))
    print(f"Average label: {tu_dataset.data.y.float().mean().item():4.2f}")

    torch.manual_seed(42)
    tu_dataset.shuffle()
    data_len = len(tu_dataset)
    train_split = 0.7
    train_dataset = tu_dataset[:int(data_len * train_split)]
    test_dataset = tu_dataset[int(data_len * train_split):]

    print(f"train: {len(train_dataset)}")
    print(f"test: {len(test_dataset)}")
    # torch.manual_seed(42)
    # tu_dataset.shuffle()
    # train_dataset = tu_dataset[:150]
    # test_dataset = tu_dataset[150:]

    graph_train_loader = geom_data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    graph_val_loader = geom_data.DataLoader(test_dataset,
                                            batch_size=64)
    # Additional loader if you want to change to a larger dataset
    graph_test_loader = geom_data.DataLoader(test_dataset, batch_size=64)

    batch = next(iter(graph_test_loader))
    print("Batch:", batch)
    print("Labels:", batch.y[:10])
    print("Batch indices:", batch.batch[:40])
    #
    model, result = train_graph_classifier(model_name="GraphConv",
                                           tu_dataset=tu_dataset,
                                           graph_train_loader=graph_train_loader,
                                           graph_val_loader=graph_val_loader,
                                           graph_test_loader=graph_test_loader,
                                           c_hidden=256,
                                           layer_name="GraphConv",
                                           num_layers=3,
                                           dp_rate_linear=0.5,
                                           dp_rate=0.0)

    print(f"Train performance: {100.0 * result['train']:4.2f}%")
    print(f"Test performance:  {100.0 * result['test']:4.2f}%")


def train(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    for data in test_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        _, predicted = torch.max(output.data, 1)
        total += data.y.size(0)
        correct += (predicted == data.y).sum().item()
    accuracy = 100 * correct / total
    return accuracy


if __name__ == '__main__':
    run()

