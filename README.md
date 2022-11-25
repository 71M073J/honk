This is a reimplementation and simplification of the [Honk](https://github.com/castorini/honk) NN for keyword recognition.

It is expected of user to have the [Speech Commands Dataset](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html) in the ```/data/speech_commands_v0.01/``` folder, otherwise this code cannot load any files.

Example sound file path: ```/data/speech_commands_v0.01/down/0a7c2a8d_nohash_0.wav```

This code makes use only of 10 words from the dataset, namely: ```yes, no, up, down, left, right, on, off, stop, go```.
Other words are by default ignored, and can be deleted.

Honk code requires as a dependency the python libraries pytorch, numpy, and the specific version of librosa```
librosa>=0.6.2,<=0.6.3```, however I have NOT tested whether it also works on newer versions.