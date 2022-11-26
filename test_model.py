import os
import wave

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utils.manage_audio import AudioPreprocessor

labelNames = "silence,unknown,yes,no,up,down,left,right,on,off,stop,go".split(",")


class VoiceCommandDataset(Dataset):
    def __init__(self, base_folder, variant="tune", transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.sound_folder = base_folder
        self.variant = variant
        self.ver = variant == "tune"
        self.lengths = []
        self.commands = []
        self.audio_processor = AudioPreprocessor()
        c = 0
        for command in os.listdir(self.sound_folder):
            if str(command) not in ["yes", "no", "up", "down", "on", "off", "left", "right", "stop", "go"]:
                continue

            command_folder = os.path.join(self.sound_folder, command)
            files = len(os.listdir(command_folder))
            self.lengths.append(0)
            self.commands.append(str(command))
            for i, file in enumerate(os.listdir(command_folder)):
                if files % 2 == 1 and i == files - 1:
                    break

                if i % 2 == self.ver:
                    self.lengths[c] += 1
            c += 1

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        n = idx
        for idn in range(len(self.lengths) + 1):
            if n < 0:
                c = self.commands[idn - 1]
                # print(len(os.listdir(os.path.join(self.sound_folder, c))),2 * n + (1 - self.ver))
                filename = os.listdir(os.path.join(self.sound_folder, c))[2 * n + (1 - self.ver)]
                fp = os.path.join(os.path.join(self.sound_folder, c), filename)
                with wave.open(fp) as f:
                    data = np.frombuffer(f.readframes(16000), dtype=np.int16) / 32768.
                    dimension_fix = np.zeros(16000)
                    dimension_fix[0:data.shape[0]] = data
                    sound = torch.from_numpy(self.audio_processor.compute_mfccs(dimension_fix).squeeze(2)).unsqueeze(
                        0)
                    sound = torch.autograd.Variable(sound, requires_grad=False)
                    label = c
                break
            else:
                n -= self.lengths[idn]
        if self.transform:
            sound = self.transform(sound)
        if self.target_transform:
            label = self.target_transform(label)
        return sound  # , label


from torch.nn import (Conv2d, Linear, MaxPool2d,
                      Module, ReLU, Sequential, Softmax)


class Classifier(Module):
    def __init__(
            self, convs: Sequential, linears: Sequential, use_softmax: bool = True
    ):
        super().__init__()
        self.convs = convs
        self.linears = linears
        self.softmax = Softmax(1) if use_softmax else Sequential()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outconvs = self.convs(inputs)
        outlins = self.linears(outconvs.view(outconvs.shape[0], -1))
        return self.softmax(outlins)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __repr__(self):
        return f'View x.unsqueeze({self.dim})'

    def forward(self, inpiut):
        '''
        Reshapes the input according to the shape saved in the unsqueeze data structure.
        '''
        out = inpiut.unsqueeze(self.dim)
        return out


class Honk(Classifier):
    def __init__(self, convs=None, linears=None):
        if convs is None:
            convs = Sequential(*(
                    [Unsqueeze(0)] +
                    [Conv2d(1, 64, (20, 8), stride=(1, 1))] +
                    [ReLU(inplace=True),
                     # Dropout(0.5)
                     ] +
                    [
                        MaxPool2d((2, 2), (2, 2)),
                        Conv2d(64, 64, (10, 4), stride=(1, 1))] +
                    [ReLU(inplace=True),
                     # Dropout(0.5)
                     ] +
                    [MaxPool2d((1, 1), (1, 1))]
                # [View()]
            ))
        if linears is None:
            linears = Sequential(Linear(26624, 12))
        super().__init__(convs, linears)


def save_bin_dataset():
    coms = "yes,no,up,down,left,right,on,off,stop,go".split(",")

    for h in ["tune", "test"]:
        d = VoiceCommandDataset(base_folder="./data/speech_commands_v0.01/", variant=h)
        alphacoms = d.commands
        idxmap = np.argsort(coms)
        print(coms, np.argsort(coms))
        print(np.sort(coms))
        print(alphacoms)
        arr = np.zeros((len(d), len(d[0]), len(d[0][0]), len(d[0][0][0])))

        arrlabel = np.zeros(len(d))
        for i, el in enumerate(d):
            arr[i] = el
            alphabetical_idx = np.sum([x < i for x in np.cumsum(d.lengths)])
            arrlabel[i] = idxmap[alphabetical_idx] + 2

            if i % 100 == 0:
                print(i, arrlabel[i])
        with open(f"{h}_input.bin", "wb") as fd:
            arr.astype("<f4").tofile(fd)  # Convert to little endian and save.

        with open(f"{h}_labels.bin", "wb") as fd:
            arrlabel.astype("int32").tofile(fd)  # Convert to little endian and save.


def save10kdataset():
    coms = "yes,no,up,down,left,right,on,off,stop,go".split(",")

    for h in ["tune", "test"]:
        d = VoiceCommandDataset(base_folder="./data/speech_commands_v0.01/", variant=h)
        alphacoms = d.commands
        idxmap = np.argsort(coms)
        print(coms, np.argsort(coms))
        print(np.sort(coms))
        print(alphacoms)
        arr = np.zeros((10000, len(d[0]), len(d[0][0]), len(d[0][0][0])))

        arrlabel = np.zeros(10000)
        ncur = 0
        lengths = np.cumsum(d.lengths)
        curlen = -1
        for i in range(10000):
            if i > 0 and i % 1000 == 0:
                curlen += 1
                ncur = lengths[curlen]
            arr[i] = d[ncur]
            alphabetical_idx = np.sum([x < ncur for x in lengths])
            arrlabel[i] = idxmap[alphabetical_idx] + 2
            ncur += 1
            if i % 100 == 0:
                print(i, arrlabel[i])

        with open(f"{h}_input10k.bin", "wb") as fd:
            arr.astype("<f4").tofile(fd)  # Convert to little endian and save.

        with open(f"{h}_labels10k.bin", "wb") as fd:
            arrlabel.astype("int32").tofile(fd)  # Convert to little endian and save.


def save500dataset():
    coms = "yes,no,up,down,left,right,on,off,stop,go".split(",")
    labelNames = "silence,unknown,yes,no,up,down,left,right,on,off,stop,go".split(",")
    for h in ["tune", "test"]:
        d = VoiceCommandDataset(base_folder="./data/speech_commands_v0.01/", variant=h)
        alphacoms = d.commands
        idxmap = np.argsort(coms)
        print(coms, np.argsort(coms))
        print(np.sort(coms))
        print(alphacoms)
        arr = np.zeros((500, len(d[0]), len(d[0][0]), len(d[0][0][0])))

        arrlabel = np.zeros(500)
        ncur = 0
        lengths = np.cumsum(d.lengths)
        curlen = -1
        for i in range(500):
            if i > 0 and i % 50 == 0:
                curlen += 1
                ncur = lengths[curlen]
            arr[i] = d[ncur]
            alphabetical_idx = np.sum([x < ncur for x in lengths])
            arrlabel[i] = idxmap[alphabetical_idx] + 2
            ncur += 1
            if i % 100 == 0:
                print(i, arrlabel[i])

        with open(f"{h}_input500.bin", "wb") as fd:
            arr.astype("<f4").tofile(fd)  # Convert to little endian and save.

        with open(f"{h}_labels500.bin", "wb") as fd:
            arrlabel.astype("int32").tofile(fd)  # Convert to little endian and save.


def eval_files(model, base_commands="./data/speech_commands_v0.01/"):
    def label(m, wav_data, audio_preprocessor):
        """Labels audio data as one of the specified trained labels

        Args:
            m: Pytorch model to get classifications from
            audio_preprocessor: class for preprocessing the data
            wav_data: The WAVE to label

        Returns:
            A (most likely label, probability) tuple
        """
        wav_data = np.frombuffer(wav_data, dtype=np.int16) / 32768.
        model_in = torch.from_numpy(audio_preprocessor.compute_mfccs(wav_data).squeeze(2)).unsqueeze(0)
        model_in = torch.autograd.Variable(model_in, requires_grad=False)
        predictions = m.forward(model_in).detach().numpy()
        return (labelNames[np.argmax(predictions)], np.max(predictions))

    preprocessor = AudioPreprocessor()
    for file in os.listdir(base_commands):
        if str(file) not in ["yes", "no", "up", "down", "on", "off", "left", "right", "stop", "go"]:
            continue

        print(file)
        true = 0
        false = 0
        base2 = os.path.join(base_commands, str(file))
        files = len(os.listdir(base2))
        cnt = -1
        for i, file2 in enumerate(os.listdir(base2)):
            # if int(i/files * 100) > cnt:
            #    cnt += 10
            #    print(i/files * 100, "%")
            fp = os.path.join(base2, file2)
            # print(fp)
            with wave.open(fp) as f:
                data = f.readframes(16000)
                lab = label(model, data, preprocessor)
                if lab[0] == file:
                    true += 1
                else:
                    false += 1
        print("true: ", int(true / (true + false) * 10000) / 100, "%")
        print("false: ", int(false / (true + false) * 10000) / 100, "%")


def eval_dataset(model, base_commands_folder="./data/speech_commands_v0.01/", variant="test"):
    d = VoiceCommandDataset(base_folder=base_commands_folder, variant=variant)
    for dataset_entry in d:
        confidences = model.forward(dataset_entry).detach().numpy()
        print(labelNames[np.argmax(confidences)], np.max(confidences))


def load_bin_dataset(dataset, dtype, shape):
    with open(dataset) as f:
        d = np.fromfile(f, dtype=dtype).reshape(shape)
        return d

def fix_weight_files(m):
    objs = [list(m.children())[0][1], list(m.children())[0][4], list(m.children())[1][0]]
    objs2 = []
    for x in objs:
        objs2 = objs2 + [x.bias.detach().numpy(), x.weight.detach().numpy()]
    for i in range(6):
        with open(f"./new_weights/weight_{i}_path.bin", "wb") as fd:
            arr = objs2[i]
            arr.astype("<f4").tofile(fd)  # Convert to little endian and save.

if __name__ == "__main__":
    base_commands = "./data/speech_commands_v0.01/"
    m = Honk()
    m.load_state_dict(torch.load("honk.pth.tar", map_location=None))
    fix_weight_files(m)
    #quit()
    eval_files(m)
    eval_dataset(m)
