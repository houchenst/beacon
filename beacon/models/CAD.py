import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))

from beacon import supernet

class SimpleCAD(supernet.SuperNet):
    def __init__(self, Args=None, DataParallelDevs=None, LatentSize=256):
        super().__init__(Args=Args)
        self.linear1 = nn.Linear(LatentSize, LatentSize)
        self.linear2 = nn.Linear(LatentSize, 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, data, embeddings):
        x = self.linear1(embeddings)
        x = self.linear2(x)
        x = torch.reshape(x, (-1, 8, 2, 2)) # b, 8, 2, 2   <-- input shape for decoder
        x = self.decoder(x)
        return x