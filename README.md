# Problem-agnostic speech embeddings for multi-speaker text-to-speech with SampleRNN
[![](https://img.shields.io/badge/publication-ISCA%20Speech%20Synthesis%20Workshop-red)](https://www.isca-speech.org/archive_v0/SSW_2019/abstracts/SSW10_O_2-3.html)
[![](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org/)
[![](https://www.codefactor.io/repository/github/davidalvarezdlt/samplernn_pase/badge)](https://www.codefactor.io/repository/github/davidalvarezdlt/samplernn_pase)
[![](https://img.shields.io/github/license/davidalvarezdlt/samplernn_pase)](https://github.com/davidalvarezdlt/samplernn_pase/blob/main/LICENSE)

This repository contains a refactored version of the code used in
["Problem-agnostic speech embeddings for multi-speaker text-to-speech with SampleRNN"](https://www.isca-speech.org/archive_v0/SSW_2019/abstracts/SSW10_O_2-3.html).
The code has been built using PyTorch Lightning, read its documentation to get
a complete overview of how this repository is structured.

**Disclaimer**: the exact version of the code used in the paper was lost. The
version published in this repository has been reconstructed and might contain
small differences.

## Preparing the data

The paper uses both VCTK and CMU Arctic speakers to train its models. Some of
the samples in those datasets do not seem to be valid, so we cleaned them
before using Merlin to extract the linguistic features. Also, notice that
the samples were trimmed and downsampled to 16 kHz. You can download the
exact dataset used in the paper (except for the PASE seeds, which have been
lost) from [this link](https://www.kaggle.com/davidalvarezdlt/samplernn-pase).

The first step is to clone this repository, install its dependencies and
``libsndfile1``:

```
git clone https://github.com/davidalvarezdlt/samplernn_pase.git
cd samplernn_pase
pip install -r requirements.txt
apt-get install libsndfile1
```

Unzip the file downloaded from the previous link inside ``./data``. The
resulting folder structure should look like this:

```
samplernn_pase/
    data/
        cmu_arctic/
        vctk/
    lightning_logs/
    samplernn_pase/
    .gitignore
    .pre-commit-config.yaml
    LICENSE
    README.md
    requirements.txt
```

## Training the model

In short, you can train the model by calling:

```
python -m samplernn_pase
```

You can modify the default parameters of the code by using CLI parameters. Get
a complete list of the available parameters by calling:

```
python -m samplernn_pase --help
```

For instance, if we want to train the model using acoustic features, with a
batch size of 32 and using one GPUs, we would call:

```
python -m samplernn_pase --conds_utterance_type acoustic --batch_size 32 --gpus 1
```

Every time you train the model, a new folder inside ``./lightning_logs``
will be created. Each folder represents a different version of the model,
containing its checkpoints and auxiliary files.

## Testing the model

You can generate random utterances from the test split and store them in
TensorBoard by calling:

```
python -m samplernn_pase --conds_utterance_type <conds_utterance_type> --test --test_checkpoint <test_checkpoint>
```

Where ``--test_checkpoint`` is a valid path to the model checkpoint that
should be used. Notice that the parameter ``--conds_utterance_type`` is
required and must be the same value used during training, as it is used to
create the model before the checkpoint is loaded.

## Citation

If you find this paper useful, please use the following citation:

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
