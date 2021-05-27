# SampleRNN in PyTorch
This repository contains a refractored version of the code used in 
[Problem-Agnostic Speech Embeddings for Multi-Speaker Text-to-Speech with SampleRNN](https://www.isca-speech.org/archive/SSW_2019/abstracts/SSW10_O_2-3.html).
This implementation is not complete, you can download the original version of the code 
[from here](https://www.dropbox.com/s/hcmqm3z0toby7hj/samplernn_pase_old.zip?dl=0). Use this version of the 
implementation if the only thing you need is the model or one of the data-related classes.

**Important note**: I have noted that, after refractoring the code, there are problems related with gradient exploding.
I do not have the time to discover what is the reasson behind this problem.

## About the data 
The paper uses both VCTK and CMU Arctic speakers to train its models. As some of the samples are invalid, I cleaned the
data before extracting the linguistic features with Merlin. Also notice that the data was trimmed and downsampled to 
16 kHz. You can download the exact set of data used in this paper 
[from this link](https://www.kaggle.com/davidalvarezdlt/samplernn-pase).

Both the data set and the loader of this implementation are not trivial to understand. While the only job of the data 
set is to return the utterances along with its features, the loader is modified so their portions are feeded
sequentially. This way of training the model is due to the fact the SampleRNN has a recurrent architecture, which 
implies forces a continuity of the hidden states between contiguous samples. For the first samples of each utterance,
this hidden state is set to be a learnable parameter.

Maybe this is no the ideal approach to train the model, but that is the way it was done in the paper and the one which
published both in this refractored version of the code and in the original one. 

## Running an experiment
The first step is to clone this repository and install its dependencies:

```
git clone https://github.com/davidalvarezdlt/samplernn_pase.git
cd samplernn_pase
pip install -r requirements.txt
```

Make sure to move the data folders `cmu_arctic/` and `vctk/` to `samplernn_pase/data`. These folders can be obtained
by uncompressing the `samplernn_pase_data.zip` file downloaded from the link of the previous section. The resulting
folder structure will look like:

```
samplernn_pase/
    data/
        cmu_arctic/
        vctk/
    experiments/
    samplernn_pase/
    config.default.json
    README.md
    requirements.txt
```
 
This refractored version is built using Skeltorch. Read its documentation to get a complete overview of how is this 
repository organized. You can create a new experiment by calling:

```
python -m samplernn_pase --experiment-name test --verbose init --config-path config.default.json
```

This will create the folder `samplernn_pase/experiments/test/` with the files required to run the experiment. Notice
that the configuration file `config.default.json` contains several important parameters related with both the data and 
the model. After the experiment has been created (it may take a while, as it is computing normalization scores for each
speaker), you can train it using:

```
python -m samplernn_pase --experiment-name test --verbose train --device cuda
```

The experiment will start training using a GPU. If you do not have one, make sure to change to `--device cpu`. The test
is runned automatically after each epoch, infering some samples from the test split.

## Citation
Please cite our paper if it has been useful for your research:

```
@inproceedings{Alvarez2019,
  author={David Álvarez and Santiago Pascual and Antonio Bonafonte},
  title={{Problem-Agnostic Speech Embeddings for Multi-Speaker Text-to-Speech with SampleRNN}},
  year=2019,
  booktitle={Proc. 10th ISCA Speech Synthesis Workshop},
  pages={35--39},
  doi={10.21437/SSW.2019-7},
  url={http://dx.doi.org/10.21437/SSW.2019-7}
}
```