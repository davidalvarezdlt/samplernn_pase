"""
Module containing the ``torch.utils.data.Dataset`` implementation of the
package.
"""
import os.path
import random
import re

import ahoproc_tools.interpolate
import ahoproc_tools.io
import numpy as np
import soundfile
import torch.utils.data


class SampleRNNPASEDataset(torch.utils.data.Dataset):
    """Implementation of the ``torch.utils.data.Dataset`` used in this package.

    Attributes:
        info: Metadata of the datasets, speakers and utterances.
        speakers_ids: Identifiers of the speakers assigned to this dataset.
        utterances_ids: Identifiers of the utterances assigned to this dataset.
        categories: Set of linguistic features that have been extracted from
            the data.
        frame_size: Number of samples predicted at every iteration.
        kwargs: Dictionary containing the CLI arguments of the execution.
    """

    def __init__(self, info, speakers_ids, utterances_ids,
                 utterances_conds_linguistic_categories, **kwargs):
        self.info = info
        self.speakers_ids = speakers_ids
        self.utterances_ids = utterances_ids
        self.categories = {
            key: list(values)
            for key, values in utterances_conds_linguistic_categories.items()
        }
        self.frame_size = np.prod(kwargs['ratios'])
        self.kwargs = kwargs

    def __getitem__(self, item):
        """Returns the samples, conditionings and metadata associated with
        the utterance at index ``self.utterances_ids[item]``.

        Args:
            item: Index of the utterance in ``self.utterances_ids``.

        Returns:
            Tuple of three positions containing:
                - A numpy array containing the waveform encoded between
                    [-1, 1].
                - A numpy array containing the conditionings of the waveform.
                - A dictionary containing the metadata of the utterance,
                    the speaker and the dataset.
        """
        utterance = self.info['utterances'][self.utterances_ids[item]]
        speaker = self.info['speakers'][utterance['speaker_id']]
        dataset = self.info['datasets'][speaker['dataset_id']]

        utterance_wav, _ = soundfile.read(os.path.join(
            dataset['wavs_folder_path'], utterance['path'] + '.wav')
        )
        utterance_conds = self._get_conds(dataset, speaker, utterance)

        utterance_wav_len, utterance_conds_len = \
            self._get_model_len(utterance_wav.shape[0])
        utterance_conds_len = min(
            utterance_conds_len, utterance_conds.shape[0]
        )
        utterance_wav_len = min(
            utterance_wav_len, utterance_conds_len * self.frame_size
        )

        utterance_wav = utterance_wav[:utterance_wav_len].astype(np.float32)
        utterance_conds = utterance_conds[:utterance_conds_len, :] \
            .astype(np.float32)
        utterance_wav = np.pad(utterance_wav, (self.frame_size, 0))

        return utterance_wav, utterance_conds, {
            'dataset': dataset, 'speaker': speaker, 'utterance': utterance
        }

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            Length of the dataset, given by the number utterances in
                ``self.utterances_ids``.
        """
        return len(self.utterances_ids)

    def shuffle_utterances(self):
        """Shuffles the order of the utterances so that they do not appear
        in the same order in the custom data loader.
        """
        random.shuffle(self.utterances_ids)

    def _get_model_len(self, utterance_wav_len_real):
        """Returns the length of both the waveform and its conditionings.

        Args:
            utterance_wav_len_real: Real length (in samples) of the waveform,
                before being cut to match model sizes.

        Returns:
            Tuple of two positions containing the length of the waveform and
            the length of the conditionings, respectively.
        """
        samples_per_forward = self.kwargs['sequence_length'] * self.frame_size
        next_seq_length_mult = \
            int((utterance_wav_len_real // samples_per_forward) *
                samples_per_forward)
        return next_seq_length_mult, int(
            next_seq_length_mult / self.frame_size
        )

    def _get_conds(self, dataset, speaker, utterance):
        """Returns the normalized conditionings associated with the utterance.

        Args:
            dataset: Metadata associated with the dataset of the utterance.
            speaker: Metadata associated with the speaker of the utterance.
            utterance: Metadata associated with the utterance.

        Returns:
            Numpy array containing the normalized conditionings of the
            utterance.
        """
        if self.kwargs['conds_utterance_type'] == 'acoustic':
            return self._get_conds_acoustic(dataset, speaker, utterance)
        elif self.kwargs['conds_utterance_type'] == 'linguistic':
            return self._get_conds_linguistic(dataset, speaker, utterance)
        else:
            conds_acoustic = self._get_conds_acoustic(
                dataset, speaker, utterance
            )
            conds_linguistic = self._get_conds_linguistic(
                dataset, speaker, utterance
            )
            utterance_conds_len = min(
                conds_acoustic.shape[0], conds_linguistic.shape[0]
            )
            return np.concatenate(
                (conds_linguistic[:utterance_conds_len, :],
                 conds_acoustic[:utterance_conds_len, -2:]), axis=1
            )

    def _get_conds_acoustic(self, dataset, speaker, utterance):
        """Returns the normalized acoustic conditionings associated with the
        utterance.

        Args:
            dataset: Metadata associated with the dataset of the utterance.
            speaker: Metadata associated with the speaker of the utterance.
            utterance: Metadata associated with the utterance.

        Returns:
            Array containing the normalized acoustic conditionings of the
            utterance.
        """
        acoustic_conds_path = os.path.join(
            dataset['conds_acoustic_folder_path'], utterance['path']
        )
        utterance_cc = ahoproc_tools.io.read_aco_file(
            acoustic_conds_path + '.cc', (-1, 40)
        )
        utterance_fv = ahoproc_tools.io.read_aco_file(
            acoustic_conds_path + '.fv', (-1,)
        )
        utterance_lf0 = ahoproc_tools.io.read_aco_file(
            acoustic_conds_path + '.lf0', (-1,)
        )
        utterance_fv, _ = ahoproc_tools.interpolate.interpolation(
            utterance_fv, 1e3
        )
        utterance_lf0, utterance_vu = ahoproc_tools.interpolate.interpolation(
            utterance_lf0, -1e10
        )
        utterance_fv = np.log(utterance_fv)
        utterance_conds = np.concatenate(
            [utterance_cc, np.expand_dims(utterance_fv, 1),
             np.expand_dims(utterance_lf0, 1),
             np.expand_dims(utterance_vu, 1)],
            axis=1
        )
        return (utterance_conds - speaker['conds_acoustic_stads'][0]) / \
            speaker['conds_acoustic_stads'][1]

    def _get_conds_linguistic(self, dataset, speaker, utterance):
        """Returns the normalized linguistic conditionings associated with the
        utterance.

        Args:
            dataset: Metadata associated with the dataset of the utterance.
            speaker: Metadata associated with the speaker of the utterance.
            utterance: Metadata associated with the utterance.

        Returns:
            Array containing the normalized linguistic conditionings of the
            utterance.
        """
        utterance_conds = []
        lab_file_path = os.path.join(
            dataset['conds_linguistic_folder_path'], utterance['path'] + '.lab'
        )
        for lab_line in SampleRNNPASEDataset.read_lab(lab_file_path):
            lab_line = np.asarray(lab_line)
            lab_line[0] = int(lab_line[1]) - int(lab_line[0])
            lab_line[2] = self.categories['phonemes'].index(lab_line[2])
            lab_line[3] = self.categories['phonemes'].index(lab_line[3])
            lab_line[4] = self.categories['phonemes'].index(lab_line[4])
            lab_line[5] = self.categories['phonemes'].index(lab_line[5])
            lab_line[6] = self.categories['phonemes'].index(lab_line[6])
            lab_line[27] = self.categories['vowels'].index(lab_line[27])
            lab_line[31] = self.categories['gpos'].index(lab_line[31])
            lab_line[33] = self.categories['gpos'].index(lab_line[33])
            lab_line[41] = self.categories['gpos'].index(lab_line[41])
            lab_line[49] = self.categories['tobi'].index(lab_line[49])
            lab_line[lab_line == 'x'] = 0
            lab_line = lab_line.astype(np.float)
            steps_n = int((lab_line[0].item() * 10E-5) / 5)
            lab_line = (lab_line - speaker['conds_linguistic_stads'][0]) / \
                speaker['conds_linguistic_stads'][1]
            utterance_ling = np.repeat(
                np.expand_dims(lab_line, 0), steps_n, axis=0
            )
            utterance_ling[:, 1] = np.linspace(0, 1, num=steps_n)
            utterance_conds.append(utterance_ling)
        return np.concatenate(utterance_conds)

    @staticmethod
    def read_lab(lab_file_path):
        """Reads a ``.lab`` file.

        Args:
            lab_file_path: Path to the ``.lab`` file.

        Returns:
            List of tuples, where each tuple contains the 55 columns of the
            ``.lab`` file.
        """
        lab_regex = ''.join([
            r'([0-9]+) ([0-9]+) ',
            r'(.+)\^(.+)-(.+)\+(.+)=(.+)@(.+)_(.+)',
            r'/A:(.+)_(.+)_(.+)',
            r'/B:(.+)-(.+)-(.+)@(.+)-(.+)&(.+)-(.+)#(.+)-(.+)\$(.+)-(.+)!(.+)-'
            r'(.+);(.+)-(.+)\|(.+)',
            r'/C:(.+)\+(.+)\+(.+)',
            r'/D:(.+)_(.+)',
            r'/E:(.+)\+(.+)@(.+)\+(.+)&(.+)\+(.+)#(.+)\+(.+)',
            r'/F:(.+)_(.+)',
            r'/G:(.+)_(.+)',
            r'/H:(.+)\=(.+)@(.+)=(.+)\|(.+)',
            r'/I:(.+)=(.+)',
            r'/J:(.+)\+(.+)-(.+)'
        ])
        lab_read_lines = []
        with open(lab_file_path) as lab_file:
            for lab_line in lab_file.readlines():
                lab_read_lines.append(
                    re.search(lab_regex, lab_line).groups())
        return lab_read_lines
