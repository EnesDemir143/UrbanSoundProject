import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor 
import torchaudio
torchaudio.set_audio_backend("soundfile")

from tqdm import tqdm
from Urbansounddataset import UrbanSoundDataset
from UrbanSoundProject.cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001

AUDIO_DIR = "UrbanSound8K/audio"
ANNOTATIONS_FILE = "UrbanSound8K/metadata/UrbanSound8K.csv"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)

        return predictions


def download_mnist_datasets():

    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )

    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )

    return train_data, validation_data


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        #Calculate Loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        #backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Loss: {loss.item()}")            


def train(model, dataloader, loss_fn, optimiser, device, epochs=30):
    for i in range(epochs):
        print(f"Epochs{i+1}")
        train_one_epoch(model, dataloader, loss_fn ,optimiser, device)
        print("-----------------------")
    print("Train is Done")
    


if __name__ == "__main__":

    #train_data, _ = download_mnist_datasets()

    if torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device = {device}")

    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectogram, SAMPLE_RATE, NUM_SAMPLES, device)
    
    print("Data downloaded")

    train_data_loader = tqdm(DataLoader(usd, batch_size=BATCH_SIZE))

    cnn = CNNNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)
 
    torch.save(cnn.state_dict(), "cnn.pth")
    print("model saved.")