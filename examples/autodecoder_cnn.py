# TODO change imports back
from beacon.supernet import SuperLoss
# from supernet import SuperLoss
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import math

# TODO: Delete this
# import torchviz
# index, data, image = next(iter(TrainDataLoader))
# output = SampleNet(data, LatVecs(index))
# loss_val = loss(output, image, embeddings=LatVecs(index))

# # torchviz.make_dot(loss_val, params=dict(SampleNet.named_parameters())).render("attached", format="png")
# vis_graph = torchviz.make_dot(loss_val, params=dict(SampleNet.named_parameters()))
# vis_graph.view()

import sys
import argparse

# TODO change imports back
from beacon.models import CAD
# from models import CAD

REG_LAMBDA = 1e-4

# Make both input and target be the same
class MNISTAutoDecoderDataset(MNIST):
    def __getitem__(self, idx):
        Image, Label = super().__getitem__(idx%100)
        return torch.tensor(idx%100, requires_grad=False), Image

    def __len__(self):
        return 1000

class ADReconstructionLoss(nn.Module):
    '''
    Computes the prediction loss for the autodecoder
    '''

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        output, _ = output
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        return self.L2(output, target)

    def L2(self, labels, predictions):
        Loss = torch.mean(torch.square(labels - predictions))
        if math.isnan(Loss) or math.isinf(Loss):
            return torch.tensor(0)
        return Loss

class ADRegularizationLoss(nn.Module):
    '''
    Computes the latent vector loss for the autodecoder
    '''

    def __init__(self, regLambda=REG_LAMBDA):
        super().__init__()
        self.regLambda = regLambda

    def forward(self, output, target):
        _, embeddings = output
        return self.computeLoss(embeddings)


    def computeLoss(self, latvecs):
        Loss = torch.mean(torch.linalg.norm(latvecs.float(), dim=-1))
        if math.isnan(Loss) or math.isinf(Loss):
            return torch.tensor(0)
        return Loss * self.regLambda

def evaluateTrain(Args, TrainData, Net, TestDevice, LatVecs):
    '''

    '''
    Net = Net.to(TestDevice)
    LatVecs = LatVecs.to(TestDevice)
    dummyData = torch.tensor([]).to(TestDevice)
    nSamples = min(Args.infer_samples, len(TrainData))
    print('[ INFO ]: Evaluating ', nSamples, ' training samples')
    otherParameters = {"Latent Vectors": LatVecs}

    for i in range(nSamples):
        Index, Image = TrainData[i]
        Index = Index.to(TestDevice)
        print(f"Index is {Index}")
        print(f"Lat Vec is {LatVecs(Index)}")
        # Embedding = LatVecs(Index)
        PredImage = Net(Index, otherParameters)[0].detach()
        plt.subplot(2, 1, 1)
        plt.imshow(Image.cpu().numpy().squeeze(), cmap='gray')
        plt.subplot(2, 1, 2)
        plt.imshow(PredImage.cpu().numpy().squeeze(), cmap='gray')
        plt.pause(1)
    
def infer(Args, TestData, Net, TestDevice):
    TestNet = Net.to(TestDevice)
    nSamples = min(Args.infer_samples, len(TestData))
    print('[ INFO ]: Testing on', nSamples, 'samples')

    for i in range(nSamples):
        Image, _ = TestData[i]
        Image = Image.to(TestDevice)
        PredImage = TestNet(Image.unsqueeze_(0)).detach()
        plt.subplot(2, 1, 1)
        plt.imshow(Image.cpu().numpy().squeeze(), cmap='gray')
        plt.subplot(2, 1, 2)
        plt.imshow(PredImage.cpu().numpy().squeeze(), cmap='gray')
        plt.pause(1)


Parser = argparse.ArgumentParser(description='Sample code that uses the beacon framework for training a simple '
                                             'autoencoder on MNIST.')
Parser.add_argument('--arch', help='Architecture to use.', choices=['SimpleCNN'], default='SimpleCNN')
Parser.add_argument('--latent-size', type=int, default=32, help="Size of latent space")
InputGroup = Parser.add_mutually_exclusive_group()
InputGroup.add_argument('--mode', help='Operation mode.', choices=['train', 'infer', 'evaluate'])
InputGroup.add_argument('--infer-samples', help='Number of samples to use during testing.', default=30, type=int)

MNISTTrans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    SampleNet = CAD.SimpleCAD(LatentSize=Args.latent_size)
    Trans = MNISTTrans

    if Args.mode == 'train':
        TrainDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        TrainData = MNISTAutoDecoderDataset(root=SampleNet.Config.Args.input_dir, train=True, download=True, transform=Trans)
        print('[ INFO ]: Data has', len(TrainData), 'samples.')
        TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=SampleNet.Config.Args.batch_size, shuffle=False, num_workers=1)
        print(f"Using batch size: {SampleNet.Config.Args.batch_size}")

        # TODO: change initialization parameters
        # LatVecs = nn.Embedding(len(TrainData), Args.latent_size, max_norm=0.001)
        LatVecs = nn.Embedding(100, Args.latent_size)
        torch.nn.init.normal_(
            LatVecs.weight.data,
            0.0,
            0.0001 / math.sqrt(Args.latent_size),
        )

        loss = SuperLoss(Losses=[ADReconstructionLoss(), ADRegularizationLoss()], Weights=[1.0,1.0], Names=["Reconstruction", "Regularization"])

        # make optimizer
        optimizer_all = torch.optim.Adam(
        [
            {
                "params": SampleNet.parameters(),
                "lr": 1e-5*SampleNet.Config.Args.batch_size,
            },
            {
                "params": LatVecs.parameters(),
                "lr": 1e-3,
            },
        ]
        )

        # Train
        SampleNet.fit(TrainDataLoader, Optimizer=optimizer_all, Objective=loss, TrainDevice=TrainDevice, OtherParameters=[LatVecs], OtherParameterNames=["Latent Vectors"])
        # print(LatVecs(torch.tensor(1)))
    elif Args.mode == 'infer':
        SampleNet.loadCheckpoint()

        TestDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        TestData = MNISTAutoDecoderDataset(root=SampleNet.Config.Args.input_dir, train=False, download=True, transform=Trans)
        print('[ INFO ]: Data has', len(TestData), 'samples.')

        infer(Args, TestData, SampleNet, TestDevice)

    elif Args.mode == 'evaluate':
        print("-"*20 + "EVALUATE" + "-"*20)
        TrainData = MNISTAutoDecoderDataset(root=SampleNet.Config.Args.input_dir, train=True, download=True, transform=Trans)
        print('[ INFO ]: Data has', len(TrainData), 'samples.')

        # TODO: change initialization parameters
        # LatVecs = nn.Embedding(len(TrainData), Args.latent_size, max_norm=0.001)
        LatVecs = nn.Embedding(100, Args.latent_size)
        torch.nn.init.normal_(
            LatVecs.weight.data,
            0.0,
            0.0001 / math.sqrt(Args.latent_size),
        )
        TestDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        SampleNet.loadCheckpoint(OtherParameters=[LatVecs], OtherParameterNames=["Latent Vectors"])

        evaluateTrain(Args, TrainData, SampleNet, TestDevice, LatVecs)


