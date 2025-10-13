import torchtext
print(torchtext.__version__)

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, Mapper
import torchtext

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

# sentences = [
#     "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
#     "Fame's a fickle friend, Harry.",
#     "It is our choices, Harry, that show what we truly are, far more than our abilities.",
#     "Soon we must all face the choice between what is right and what is easy.",
#     "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
#     "You are awesome!"
# ]

# class CustomDataset(Dataset):
#     def __init__(self, sentences):
#         self.sentences = sentences
#
#     def __len__(self):
#         return len(self.sentences)
#
#     def __getitem__(self, idx):
#         return self.sentences[idx]
#
# custom_dataset = CustomDataset(sentences)
#
# batch_size = 2
#
# dataloader = DataLoader(custom_dataset, batch_size = batch_size, shuffle = True)
#
# for batch in dataloader:
#     print(batch)

# class CustomDataset(Dataset):
#     def __init__(self,sentences,tokenizer, vocab):
#         self.sentences = sentences
#         self.tokenizer = tokenizer
#         self.vocab = vocab
#
#     def __len__(self):
#         return len(self.sentences)
#
#
#     def __getitem__(self, idx):
#         tokens = self.tokenizer(self.sentences[idx])
#         tensor_indices = [self.vocab[token] for token in tokens]
#         return torch.tensor(tensor_indices)


# tokenizer = get_tokenizer('basic_english')
#
# vocab = build_vocab_from_iterator(map(tokenizer, sentences))

# custom_dataset = CustomDataset(sentences, tokenizer, vocab)
#
# print('Custom Dataset Length', len(custom_dataset))
# print('Sample Items:')
#
# for i in range(6):
#     sample_item = custom_dataset[i]
#     print(f"Item {i + 1}: {sample_item}")

# Create an instance of your custom data set

# def collate_fn(batch):
#     # Pad sequences within the batch to have equal lengths
#     padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
#     return padded_batch
#
# custom_dataset = CustomDataset(sentences, tokenizer, vocab)
#
# # Define batch size
# batch_size = 2
#
# # Create a data loader
# dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn)
#
# # Iterate through the data loader
# for batch in dataloader:
#     print(batch)
#
# for batch in dataloader:
#     for row in batch:
#         for idx in row:
#             words = [vocab.get_itos()[idx] for idx in row]
#         print(words)

corpus = [
    "Ceci est une phrase.",
    "C'est un autre exemple de phrase.",
    "Voici une troisième phrase.",
    "Il fait beau aujourd'hui.",
    "J'aime beaucoup la cuisine française.",
    "Quel est ton plat préféré ?",
    "Je t'adore.",
    "Bon appétit !",
    "Je suis en train d'apprendre le français.",
    "Nous devons partir tôt demain matin.",
    "Je suis heureux.",
    "Le film était vraiment captivant !",
    "Je suis là.",
    "Je ne sais pas.",
    "Je suis fatigué après une longue journée de travail.",
    "Est-ce que tu as des projets pour le week-end ?",
    "Je vais chez le médecin cet après-midi.",
    "La musique adoucit les mœurs.",
    "Je dois acheter du pain et du lait.",
    "Il y a beaucoup de monde dans cette ville.",
    "Merci beaucoup !",
    "Au revoir !",
    "Je suis ravi de vous rencontrer enfin !",
    "Les vacances sont toujours trop courtes.",
    "Je suis en retard.",
    "Félicitations pour ton nouveau travail !",
    "Je suis désolé, je ne peux pas venir à la réunion.",
    "À quelle heure est le prochain train ?",
    "Bonjour !",
    "C'est génial !"
]


class CustomDataset(Dataset):
    def __init__(self,sentences, tokenizer, vocab):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

custom_dataset = CustomDataset(corpus)


def collate_fn(batch):
    tensor_batch = []
    for sample in batch:
        tokens = tokenizer(sample)
        tensor_batch.append(torch.tensor([vocab[token] for token in tokens]))

    padded_batch = pad_sequence(tensor_batch, batch_first=True)
    return padded_batch





