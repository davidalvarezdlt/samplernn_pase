import ahoproc_tools.io
import ahoproc_tools.interpolate
import numpy as np
import os.path
import random
import samplernn_pase.utils as utils
import soundfile
import torch.utils.data


class SampleRNNPASEDataset(torch.utils.data.Dataset):
    datasets_info = None
    speakers_info = None
    utterances_info = None
    speakers_ids = None
    utterances_ids = None
    conds_utterance_type = None
    conds_utterance_linguistic_categories = None
    sequence_length = None
    frame_size = None
    split = None

    def __init__(self, datasets_info, speakers_info, utterances_info, speakers_ids, utterances_ids,
                 conds_utterance_type, utterances_conds_linguistic_categories, sequence_length, frame_layers_ratios,
                 split):
        self.datasets_info = datasets_info
        self.speakers_info = speakers_info
        self.utterances_info = utterances_info
        self.speakers_ids = speakers_ids
        self.utterances_ids = utterances_ids
        self.conds_utterance_type = conds_utterance_type
        self.conds_utterance_linguistic_categories = {
            key: list(values) for key, values in utterances_conds_linguistic_categories.items()
        }
        self.sequence_length = sequence_length
        self.frame_size = np.prod(frame_layers_ratios)
        self.split = split

    def __getitem__(self, item):
        utterance = self.utterances_info[self.utterances_ids[item]]
        speaker = self.speakers_info[utterance['speaker_id']]
        dataset = self.datasets_info[speaker['dataset_id']]
        utterance_wav, _ = soundfile.read(os.path.join(dataset['wavs_folder_path'], utterance['path'] + '.wav'))
        utterance_conds = self._get_conds(dataset, speaker, utterance)
        utterance_wav_len, utterance_conds_len = self._get_model_len(utterance_wav.shape[0])
        utterance_conds_len = min(utterance_conds_len, utterance_conds.shape[0])
        utterance_wav_len = min(utterance_wav_len, utterance_conds_len * self.frame_size)
        utterance_wav = utterance_wav[:utterance_wav_len].astype(np.float32)
        utterance_conds = utterance_conds[:utterance_conds_len, :].astype(np.float32)
        utterance_wav = np.pad(utterance_wav, (self.frame_size, 0))
        return utterance_wav, utterance_conds, {'dataset': dataset, 'speaker': speaker, 'utterance': utterance}

    def __len__(self):
        return len(self.utterances_ids)

    def shuffle_utterances(self):
        random.shuffle(self.utterances_ids)

    def _get_model_len(self, utterance_wav_len_real):
        samples_per_forward = self.sequence_length * self.frame_size
        next_seq_length_mult = int((utterance_wav_len_real // samples_per_forward) * samples_per_forward)
        return next_seq_length_mult, int(next_seq_length_mult / self.frame_size)

    def _get_conds(self, dataset, speaker, utterance):
        if self.conds_utterance_type == 'acoustic':
            return self._get_conds_acoustic(dataset, speaker, utterance)
        elif self.conds_utterance_type == 'linguistic':
            return self._get_conds_linguistic(dataset, speaker, utterance)
        else:
            conds_acoustic = self._get_conds_acoustic(dataset, speaker, utterance)
            conds_linguistic = self._get_conds_linguistic(dataset, speaker, utterance)
            utterance_conds_len = min(conds_acoustic.shape[0], conds_linguistic.shape[0])
            return np.concatenate(
                (conds_linguistic[:utterance_conds_len, :], conds_acoustic[:utterance_conds_len, -2:]), axis=1
            )

    def _get_conds_acoustic(self, dataset, speaker, utterance):
        acoustic_conds_path = os.path.join(dataset['conds_utterance']['acoustic_folder_path'], utterance['path'])
        utterance_cc = ahoproc_tools.io.read_aco_file(acoustic_conds_path + '.cc', (-1, 40))
        utterance_fv = ahoproc_tools.io.read_aco_file(acoustic_conds_path + '.fv', (-1,))
        utterance_lf0 = ahoproc_tools.io.read_aco_file(acoustic_conds_path + '.lf0', (-1,))
        utterance_fv, _ = ahoproc_tools.interpolate.interpolation(utterance_fv, 1e3)
        utterance_lf0, utterance_vu = ahoproc_tools.interpolate.interpolation(utterance_lf0, -1e10)
        utterance_fv = np.log(utterance_fv)
        utterance_conds = np.concatenate(
            [utterance_cc, np.expand_dims(utterance_fv, 1), np.expand_dims(utterance_lf0, 1),
             np.expand_dims(utterance_vu, 1)],
            axis=1
        )
        return (utterance_conds - speaker['conds_acoustic_stads'][0]) / speaker['conds_acoustic_stads'][1]

    def _get_conds_linguistic(self, dataset, speaker, utterance):
        utterance_conds = []
        lab_file_path = os.path.join(dataset['conds_utterance']['linguistic_folder_path'], utterance['path'] + '.lab')
        for lab_line in utils.read_lab(lab_file_path):
            lab_line = np.asarray(lab_line)
            lab_line[0] = int(lab_line[1]) - int(lab_line[0])
            lab_line[2] = self.conds_utterance_linguistic_categories['phonemes'].index(lab_line[2])
            lab_line[3] = self.conds_utterance_linguistic_categories['phonemes'].index(lab_line[3])
            lab_line[4] = self.conds_utterance_linguistic_categories['phonemes'].index(lab_line[4])
            lab_line[5] = self.conds_utterance_linguistic_categories['phonemes'].index(lab_line[5])
            lab_line[6] = self.conds_utterance_linguistic_categories['phonemes'].index(lab_line[6])
            lab_line[27] = self.conds_utterance_linguistic_categories['vowels'].index(lab_line[27])
            lab_line[31] = self.conds_utterance_linguistic_categories['gpos'].index(lab_line[31])
            lab_line[33] = self.conds_utterance_linguistic_categories['gpos'].index(lab_line[33])
            lab_line[41] = self.conds_utterance_linguistic_categories['gpos'].index(lab_line[41])
            lab_line[49] = self.conds_utterance_linguistic_categories['tobi'].index(lab_line[49])
            lab_line[lab_line == 'x'] = 0
            lab_line = lab_line.astype(np.float)
            steps_n = int((lab_line[0].item() * 10E-5) / 5)
            lab_line = (lab_line - speaker['conds_linguistic_stads'][0]) / speaker['conds_linguistic_stads'][1]
            utterance_ling = np.repeat(np.expand_dims(lab_line, 0), steps_n, axis=0)
            utterance_ling[:, 1] = np.linspace(0, 1, num=steps_n)
            utterance_conds.append(utterance_ling)
        return np.concatenate(utterance_conds)
