from __future__ import print_function
import torch
import torchvision.transforms as transforms

import os
import sys
from datasets import TextDataset
from torch.utils.data import DataLoader
from trainer import FineGAN_trainer
from model import RNN_ENCODER, CNN_ENCODER

if __name__ == "__main__":
    BATCH_SIZE = 16
    image_transform = transforms.Compose(
        [
            transforms.Scale(int(128 * 76 / 64)),
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip()
        ]
    )
    dataset = TextDataset('../data/flowers', split='train', transform=image_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=8)

    text_encoder = RNN_ENCODER(ntoken=5428, nhidden=256)
    state_dict = torch.load('../models/flowers/text_encoder200.pth')
    text_encoder.load_state_dict(state_dict)

    image_encoder = CNN_ENCODER()
    state_dict = torch.load('../models/flowers/image_encoder200.pth')
    image_encoder.load_state_dict(state_dict)

    trainer = FineGAN_trainer(output_dir='../output/lab', data_loader=dataloader, imsize=128)
    trainer.train(text_encoder, image_encoder, dataset.ixtoword)
