import json
import os
import wave

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import librosa
from utils.manage_audio import AudioPreprocessor
import scipy.io.wavfile

labelNames = "silence,unknown,yes,no,up,down,left,right,on,off,stop,go".split(",")


class VoiceCommandDataset(Dataset):
    def __init__(self, base_folder, variant="tune", transform=None, target_transform=None, preprocessed=True):
        self.preprocessed = preprocessed
        self.transform = transform
        self.target_transform = target_transform
        self.sound_folder = base_folder
        self.variant = variant
        self.ver = (variant == "tune") + (variant == "all") * 2
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
                elif self.ver == 2:
                    self.lengths[c] += 1
            c += 1

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        if self.ver != 2:
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
                        if self.preprocessed:
                            sound = torch.from_numpy(
                                self.audio_processor.compute_mfccs(dimension_fix).squeeze(2)).unsqueeze(
                                0)
                            sound = torch.autograd.Variable(sound, requires_grad=False)
                        else:
                            sound = dimension_fix
                        label = c
                    break
                else:
                    n -= self.lengths[idn]
        else:
            idxs = np.cumsum(np.array(self.lengths)) - idx
            idn = np.argmax(idxs > 0)
            c = self.commands[idn]
            idxs2 = np.zeros(len(idxs) + 1, dtype=np.int32)
            idxs2[1:] = idxs
            # print(len(os.listdir(os.path.join(self.sound_folder, c))),2 * n + (1 - self.ver))
            filename = os.listdir(os.path.join(self.sound_folder, c))[idx - idxs2[idn]]
            fp = os.path.join(os.path.join(self.sound_folder, c), filename)
            with wave.open(fp) as f:
                data = np.frombuffer(f.readframes(16000), dtype=np.int16)
                dimension_fix = np.zeros(16000)
                dimension_fix[0:data.shape[0]] = data
                sound = dimension_fix
                label = c

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


def fix_weight_files(m, prefix="./new_weights"):
    objs = [list(m.children())[0][1], list(m.children())[0][4], list(m.children())[1][0]]
    objs2 = []
    for x in objs:
        objs2 = objs2 + [x.bias.detach().numpy(), x.weight.detach().numpy()]
    for i in range(6):
        with open(f"{prefix}/weight_{i}_path.bin", "wb") as fd:
            arr = objs2[i] if i < 5 else objs2[i].T
            arr.astype("<f4").tofile(fd)  # Convert to little endian and save.


def test_weights(m):
    objs = [list(m.children())[0][1], list(m.children())[0][4], list(m.children())[1][0]]
    objs2 = []
    objs3 = []
    np.set_printoptions(threshold=123123123123)
    for x in objs:
        objs2 = objs2 + [x.bias.detach().numpy().shape, x.weight.detach().numpy().shape]
        objs3 = objs3 + [x.bias.detach().numpy(), x.weight.detach().numpy()]
    for i in range(0, 6, 1):
        with open(f"./new_weights/weight_{i}_path.bin", "rb") as fd:
            arr = np.fromfile(fd, dtype="<f4").reshape(objs2[i])  # Convert to little endian and save.
        with open(f"./new_weights/old_weights/weight_{i}_path.bin", "rb") as fd:
            arr2 = np.fromfile(fd, dtype="<f4").reshape(objs2[i])  # Convert to little endian and save.
        with open(f"E:/Users/timotej999/Documents/HPVM/hpvm-tuning-android/singularity/android-approxhpvm"
                  f"-demo_mobilenet/app/src/main/assets/models/honk/weight_{i}_path.bin", "rb") as fd:
            arr3 = np.fromfile(fd, dtype="<f4").reshape(objs2[i])  # Convert to little endian and save.
        test = objs3[i] - arr3
        # print(test)
        print(test.sum())
    # m.eval()
    print(m.forward(torch.Tensor(np.zeros((1, 101, 40)))))
    with open("test.txt") as f:
        inp = f.read().replace("arrayOf", "").replace("floatArrayOf", "").replace("F", "").replace("(", "[").replace(
            ")", "]").replace("\n", "")
        print(inp)
        inp = json.loads(inp)
        inp = np.array(inp)
        print(m.forward(torch.Tensor(inp).unsqueeze(0)))


def arr_to_kotlin_code(arr, filename):
    with open(filename, "w") as f:
        print(",".join(str(arr).split()).replace("],[", "F],\n[").replace(",", "F,").replace("]F,", "],")
              .replace("[[[", "arrayOf([").replace("[", "floatArrayOf(").replace("]]]", "F)\n)").replace("]", ")")
              .replace("(f", "(\nf"), file=f)


talking = None
random = None
gauss = None

def add_noise_to_dataset_and_save():
    def add_noise_to_sound(sound, type, strength):
        sound = sound / np.abs(sound).max()
        if type == "white":
            global random
            if random is None:
                random = ((np.random.random(16000) - 0.5) * 2)
                random = random / np.abs(random).max()
            sound = sound + (random * (strength / 100.))
        elif type == "gaussian":
            global gauss
            if gauss is None:
                gauss = np.random.normal(0, 1, 16000)
                gauss = gauss / np.abs(gauss).max()
            sound = sound + (gauss * (strength / 100.))
        elif type == "talking":
            global talking
            if not os.path.exists("talking.wav"):
                y, s = librosa.load("Small Crowd Talking Ambience [TubeRipper.com].wav", sr=16000, offset=44,
                                    duration=1)
                with wave.open("talking.wav", "w") as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(16000)
                    f.writeframes((y * (2 ** 15)).astype(np.int16))

            elif talking is None:
                with wave.open("talking.wav") as f:
                    talking = np.frombuffer(f.readframes(16000), dtype=np.int16)
            sound = sound + (talking / 32768.) * (strength / 100.)
        sound = sound / np.abs(sound).max()
        return sound

    base_commands = "./data/speech_commands_v0.01/"
    variant = "tune"
    for variant in ["test", "tune"]:
        d = VoiceCommandDataset(base_commands, variant=variant, preprocessed=False)
        arr = np.zeros((len(d), 1, 101, 40), dtype=np.int16)
        for strength in [5, 10, 25, 50, 75]:
            for noise in ["talking", "white", "gaussian", ]:
                percent = -1
                print(f"Noise type:{noise}, strength: {strength}%")
                for i, s in enumerate(d):
                    h = int((i / len(d)) * 100)
                    if h != percent:
                        percent = h
                        print(f"\r{h}% done in this iteration.", end="")
                    sound = add_noise_to_sound(s, noise, strength)
                    sound = torch.from_numpy(d.audio_processor.compute_mfccs(sound).squeeze(2)).unsqueeze(0).numpy()
                    arr[i] = sound
                print("100%")
                arr.astype("<f4").tofile(f"./new_datasets/dataset_{noise}_{strength}_{variant}.bin")
        break

def plot_noise_response():
    cnt = 1
    data = np.zeros((3, 5, 51))
    for j, noise_type in enumerate(["gaussian", "talking", "white"]):
        for i, strength in enumerate([5, 10, 25, 50, 75]):
            with open(f"./new_datasets/{cnt}.txt", "r") as f:
                baseline = float(f.readline().split()[-1])
                others = []
                for line in f:
                    others.append(float(line.split()[3][:-1]))
                others = np.array(others)
                data[j,i, :-1] = others
                data[j,i, -1] = baseline
                cnt += 3
        cnt = (cnt + 1) % 15
    datas = data.sum(axis=1).sum(axis=0)
    inds = np.argsort(datas)
    sorted_data = datas[inds]

    generate_small_confs(inds[-10:])

    #for asda in range(3):
    for i in range(51):
        plt.plot([5,10,25,50,75], data[:,:,i].T)
    plt.xticks([5,10,25,50,75], [5,10,25,50,75])
    plt.xlabel("% signal to noise ratio")
    plt.ylabel("% accuracy")
    plt.show()

def generate_small_confs(indexes):
    new_confs = []
    with open("./confs.txt", "r") as f:
        start = f.readline()
        curr_conf = []
        for line in f:
            curr_conf.append(line)
            if line.startswith("-"):
                new_confs.append(curr_conf)
                curr_conf = []
                continue

    old_confs = []
    with open("./confs_old.txt", "r") as f:
        start1 = f.readline()
        curr_conf = []
        for line in f:
            curr_conf.append(line)
            if line.startswith("-"):
                old_confs.append(curr_conf)
                curr_conf = []
                continue


    new_confs = sorted(new_confs, key=lambda x: x[1].split()[1])
    indexes2 = [int(x[1].split()[0][4:]) for x in new_confs]
    print(indexes2)
    old_confs = np.array(old_confs)[indexes2]
    with open("./new_old_confs.txt", "w") as f:
        f.write(start)
        cnt = 0
        for line in old_confs[0]:
            if line.startswith("conf"):
                l = line.split()
                l[0] = f"conf{cnt}"
                f.write(" ".join(l) + "\n")
            else:
                f.write(line)
        for i, conf in enumerate(old_confs):
            if i in indexes:
                cnt += 1
                for line in conf:

                    if line.startswith("conf"):
                        l = line.split()
                        l[0] = f"conf{cnt}"
                        f.write(" ".join(l) + "\n")
                    else:
                        f.write(line)


    with open("./new_confs.txt", "w") as f:
        f.write(start)
        cnt = 0
        for line in new_confs[0]:
            if line.startswith("conf"):
                l = line.split()
                l[0] = f"conf{cnt}"
                f.write(" ".join(l) + "\n")
            else:
                f.write(line)
        for i, conf in enumerate(new_confs):
            if i in indexes:
                cnt += 1
                for line in conf:

                    if line.startswith("conf"):
                        l = line.split()
                        l[0] = f"conf{cnt}"
                        f.write(" ".join(l) + "\n")
                    else:
                        f.write(line)

if __name__ == "__main__":
    plot_noise_response()
    quit()
    add_noise_to_dataset_and_save()
    quit()

    base_commands = "./data/speech_commands_v0.01/"
    d = VoiceCommandDataset(base_commands, "tune")
    torch.set_printoptions(profile="full")
    np.set_printoptions(threshold=123123123123, precision=123)
    print(d.commands, np.sort(d.commands), np.argsort(d.commands))
    m = Honk()
    m.load_state_dict(torch.load("honk.pth.tar", map_location=None))
    fix_weight_files(m, "./new_weights_hopefully")
    # for h in range(1):
    #    for i in range(3):
    #        print(i + np.cumsum(d.lengths)[h], h)#
    #        confidences = m.forward(d[i + np.cumsum(d.lengths)[h]]).detach().numpy()
    #        print(labelNames[np.argmax(confidences)], np.max(confidences), confidences)

    h = 0
    i = 1 + np.cumsum(d.lengths)[h]
    arr_to_kotlin_code(d[i].detach().numpy(), "./kotlinarr.txt")
    # plt.imshow(d[i])
    confidences = m.forward(d[i]).detach().numpy()
    print(labelNames[np.argmax(confidences)], np.max(confidences), confidences)
    print()
    quit()
    fix_weight_files(m, "./new_weights/test_dir")
    test_weights(m)
    quit()
    fix_weight_files(m)
    # quit()
    eval_files(m)
    eval_dataset(m)
