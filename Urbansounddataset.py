from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os 
import torch
import torchaudio

AUDIO_DIR = "UrbanSound8K/audio"
ANNOTATIONS_FILE = "UrbanSound8K/metadata/UrbanSound8K.csv"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.device = device
        self.aanotations = pd.read_csv(annotations_file)
        self.audior_dir = audio_dir
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.aanotations)


    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)  
            #[1, 1, 1] -> [1, 1, 1, 0, 0] (0, 2) ise
            #0 burda left padding, 2 burda right padding sayısı
            # (1, 1, 2, 2) olursa ilk 2 son dimemtiona diğer ikisi sondan bir önceki dimentiona!!!
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
            
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal  

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
        
    
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.aanotations.iloc[index, 5]}" # 5 is index of fold in csv
        path = os.path.join(self.audior_dir, fold, self.aanotations.iloc[index, 0])
        return path
    
    def _get_audio_sample_label(self, index):
        return self.aanotations.iloc[index, 6]        
 

if __name__ == "__main__":

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

    print(f"There are {len(usd)} samples in dataset."),

    signal, label = usd[0]

    print(signal.shape)
