"""
Module containing the ``pytorch_lightning.LightningDataModule`` implementation
of the package.
"""
import concurrent.futures
import csv
import operator
import os.path
import pickle
import random

import ahoproc_tools.interpolate
import ahoproc_tools.io
import numpy as np
import progressbar
import pytorch_lightning as pl
import soundfile
import torch.utils.data

from .dataset import SampleRNNPASEDataset
from .loader import SampleRNNPASELoader


class SampleRNNPASEData(pl.LightningDataModule):
    """Implementation of the ``pytorch_lightning.LightningDataModule`` used in
    this package.

    Attributes:
        info: Metadata of the datasets, speakers and utterances.
        categories: Set of linguistic features that have been extracted from
            the data.
        speakers_ids: Identifiers of the speakers assigned to both modeling
            and adaptation splits.
        utterances_ids: Identifiers of the utterances assigned to the
            training, validation and test splits of both modeling and
            adaptation.
        kwargs: Dictionary containing the CLI arguments of the execution.
    """
    PHONEME_FEATURES_INDEXES = [2, 3, 4, 5, 6]
    VOWEL_FEATURES_INDEXES = [27]
    GPOS_FEATURES_INDEXES = [31, 33, 41]
    TOBI_FEATURES_INDEXES = [49]
    REAL_FEATURES_INDEXES = [
        7, 8, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30, 32,
        34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53,
        54
    ]

    def __init__(self, **kwargs):
        """Creates a new instance of ``SampleRNNPASEData``.

        Args:
            **kwargs: Dictionary containing the CLI arguments
                required to create an instance of ``SampleRNNPASEData``.
        """
        super().__init__()
        self.info = {'datasets': {}, 'speakers': {}, 'utterances': {}}
        self.categories = {
            'phonemes': set(), 'vowels': set(), 'gpos': set(), 'tobi': set()
        }
        self.speakers_ids = {'modeling': [], 'adaptation': []}
        self.utterances_ids = {
            'modeling': {'train': [], 'val': [], 'test': []},
            'adaptation': {'train': [], 'val': [], 'test': []}
        }
        self.kwargs = kwargs

    def prepare_data(self):
        """Prepares the data used to train the model.

        Fills all class parameters, including metadata and data allocation
        parameters.
        """
        if os.path.exists(self.kwargs['data_ckpt_path']):
            with open(self.kwargs['data_ckpt_path'], 'rb') as data_ckpt:
                self.info, self.categories, self.speakers_ids, \
                    self.utterances_ids = pickle.load(data_ckpt)
        else:
            self._load_datasets_info()
            for dataset_id, _ in self.info['datasets'].items():
                self._load_speakers_info(dataset_id)

            bar = progressbar.ProgressBar(
                max_value=len(self.info['speakers'].items())
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as \
                    executor:
                for speaker_id, _ in self.info['speakers'].items():
                    executor.submit(self._load_speaker_data, speaker_id, bar)

            self._allocate_speakers_ids()
            self._allocate_utterances_ids()

            with open(self.kwargs['data_ckpt_path'], 'wb') as data_ckpt:
                pickle.dump((self.info, self.categories, self.speakers_ids,
                             self.utterances_ids), data_ckpt)

    def _load_datasets_info(self):
        """Fills the parameter ``self.info['datasets']``.
        """
        self.info['datasets']['vctk'] = {
            'index': len(self.info['datasets']),
            'name': 'vctk',
            'speakers_prefix': 'vctk_',
            'info_file_path': os.path.join(
                self.kwargs['data_path'], 'vctk', 'speaker-info.txt'
            ),
            'wavs_folder_path': os.path.join(
                self.kwargs['data_path'], 'vctk', 'wav16_trimmed'
            ),
            'conds_acoustic_folder_path': os.path.join(
                self.kwargs['data_path'], 'vctk',
                'wav16_trimmed_conds_acoustic'
            ),
            'conds_linguistic_folder_path': os.path.join(
                self.kwargs['data_path'], 'vctk',
                'wav16_trimmed_conds_linguistic'
            ),
        }
        self.info['datasets']['cmu_arctic'] = {
            'index': len(self.info['datasets']),
            'name': 'cmu_artic',
            'speakers_prefix': 'cmu_',
            'info_file_path': os.path.join(
                self.kwargs['data_path'], 'cmu_arctic', 'speaker-info.txt'
            ),
            'wavs_folder_path': os.path.join(
                self.kwargs['data_path'], 'cmu_arctic', 'wav16_trimmed'
            ),
            'conds_acoustic_folder_path': os.path.join(
                self.kwargs['data_path'], 'cmu_arctic',
                'wav16_trimmed_conds_acoustic'
            ),
            'conds_linguistic_folder_path': os.path.join(
                self.kwargs['data_path'], 'cmu_arctic',
                'wav16_trimmed_conds_linguistic'
            )
        }

    def _load_speakers_info(self, dataset_id):
        """Fills the parameter ``self.info['speakers']`` for the speakers of
        ``dataset_id``.

        Args:
            dataset_id: Identifier of the dataset from which to load speakers
            information.
        """
        dataset = self.info['datasets'][dataset_id]
        info_list = self._read_dataset_info_file_vctk() \
            if dataset_id == 'vctk' else self._read_dataset_info_file_cmu()
        for speaker_name, speaker_gender in info_list:
            self.info['speakers'][
                dataset['speakers_prefix'] + speaker_name] = {
                'index': len(self.info['speakers']),
                'dataset_id': dataset_id,
                'name': speaker_name,
                'gender': speaker_gender,
                'wavs_len': 0,
                'conds_acoustic_stads': (0, 0),
                'conds_linguistic_stads': (0, 0)
            }

    def _read_dataset_info_file_vctk(self):
        """Reads the information file of the VCTK dataset.

        Returns:
            List of tuples, where each tuple contains the speaker identifier
                and its gender.
        """
        info_list = []
        info_file_path = self.info['datasets']['vctk']['info_file_path']
        with open(info_file_path, 'r') as info_file:
            speakers_info = csv.reader(info_file, delimiter=' ')
            speakers_info_it = iter(speakers_info)
            next(speakers_info_it)
            for speakers_info_row in speakers_info_it:
                info_list.append(
                    ('p' + speakers_info_row[0], speakers_info_row[4])
                )
        return info_list

    def _read_dataset_info_file_cmu(self):
        """Reads the information file of the CMU Arctic dataset.

        Returns:
            List of tuples, where each tuple contains the speaker identifier
                and its gender.
        """
        info_list = []
        info_file_path = self.info['datasets']['cmu_arctic']['info_file_path']
        with open(info_file_path, 'r') as info_file:
            speakers_info = csv.reader(info_file, delimiter=',')
            speakers_info_it = iter(speakers_info)
            next(speakers_info_it)
            for speakers_info_row in speakers_info_it:
                info_list.append((speakers_info_row[0], speakers_info_row[1]))
        return info_list

    def _load_speaker_data(self, speaker_id, bar):
        """Fills the parameters ``self.info['utterances']``,

        Args:
            speaker_id: Identifier of the speaker from which to load utterances
                information.
            bar: Progress bar used to track progress.
        """
        self._load_utterances_info(speaker_id)
        self._load_conds_acoustic_stats(speaker_id)
        self._load_conds_linguistic_stats(speaker_id)
        bar.update(max(bar.value, bar.value + 1))

    def _load_utterances_info(self, speaker_id):
        """Fills the parameter ``self.info['utterances']`` for the
        utterances of ``speaker_id``.

        Args:
            speaker_id: Identifier of the speaker from which to load utterances
                information.
        """
        speaker = self.info['speakers'][speaker_id]
        dataset = self.info['datasets'][speaker['dataset_id']]
        speakers_wavs_folder_path = os.path.join(
            dataset['wavs_folder_path'], speaker['name']
        )

        for _, _, utterances_names in os.walk(speakers_wavs_folder_path):
            for utterance_name in utterances_names:
                if utterance_name[-4:] != '.wav':
                    continue

                utterance_id = speaker_id + '-' + utterance_name[:-4]
                utterance, utterance_sf = soundfile.read(
                    os.path.join(speakers_wavs_folder_path, utterance_name)
                )
                self.info['utterances'][utterance_id] = {
                    'index': len(self.info['utterances']),
                    'speaker_id': speaker_id,
                    'name': utterance_name[:-4],
                    'wav_len': utterance.size / utterance_sf,
                    'path': speaker['name'] + os.sep + utterance_name[:-4]
                }
                speaker['wavs_len'] += \
                    self.info['utterances'][utterance_id]['wav_len']

    def _load_conds_acoustic_stats(self, speaker_id):
        """Fills the parameter
        ``self.info['speakers'][speaker_id]['conds_acoustic_stads']`` for the
        speaker with ID ``speaker_id``.

        Args:
            speaker_id: Identifier of the speaker from which to load utterances
                information.
        """
        speaker = self.info['speakers'][speaker_id]
        dataset = self.info['datasets'][speaker['dataset_id']]
        speaker_conds_acoustic_dir = os.path.join(
            dataset['conds_acoustic_folder_path'], speaker['name']
        )
        utterances_aux = None

        for _, _, utterances_names in os.walk(speaker_conds_acoustic_dir):
            for utterance_name in utterances_names:
                if utterance_name[-3:] != '.cc':
                    continue

                utterance_cc_path = os.path.join(
                    speaker_conds_acoustic_dir, utterance_name
                )
                utterance_cc = ahoproc_tools.io.read_aco_file(
                    utterance_cc_path, (-1, 40)
                )

                utterance_fv_path = os.path.join(
                    speaker_conds_acoustic_dir, utterance_name[:-3] + '.fv'
                )
                utterance_fv = ahoproc_tools.io.read_aco_file(
                    utterance_fv_path, (-1)
                )

                utterance_lf0_path = os.path.join(
                    speaker_conds_acoustic_dir, utterance_name[:-3] + '.lf0'
                )
                utterance_lf0 = ahoproc_tools.io.read_aco_file(
                    utterance_lf0_path, (-1)
                )

                # Interpolate both FV and LF0 and obtain VU
                utterance_fv, _ = ahoproc_tools.interpolate \
                    .interpolation(utterance_fv, 1e3)
                utterance_lf0, utterance_vu = ahoproc_tools.interpolate. \
                    interpolation(utterance_lf0, -1e10)

                # Compute LOG(FV)
                utterance_fv = np.log(utterance_fv)

                # Merge all the conds, set 1 in the last position so we get
                # MEAN=0 & STD=0 for VU bool conditionings
                utterance_conds = np.concatenate([
                    utterance_cc,
                    np.expand_dims(utterance_fv, 1),
                    np.expand_dims(utterance_lf0, 1),
                    np.zeros((utterance_cc.shape[0], 1))
                ], axis=1)

                # Store FEATURES to obtain the MEAN and STD
                if utterances_aux is None:
                    utterances_aux = utterance_conds
                else:
                    utterances_aux = np.concatenate(
                        [utterances_aux, utterance_conds]
                    )

        utterances_means = np.mean(utterances_aux, axis=0)
        utterances_stds = np.std(utterances_aux, axis=0)

        # Set the STD=1 for the bool conditioning
        utterances_stds[-1] = 1

        # Store MEAN and STD in the information of the speaker
        speaker['conds_acoustic_stads'] = (
            utterances_means.tolist(), utterances_stds.tolist()
        )

    def _load_conds_linguistic_stats(self, speaker_id):
        """Fills the parameter
        ``self.info['speakers'][speaker_id]['conds_linguistic_stads']`` for the
        speaker with ID ``speaker_id``.

        Args:
            speaker_id: Identifier of the speaker from which to load utterances
                information.
        """
        speaker = self.info['speakers'][speaker_id]
        dataset = self.info['datasets'][speaker['dataset_id']]
        speaker_conds_linguistic_dir = os.path.join(
            dataset['conds_linguistic_folder_path'], speaker['name']
        )
        conds_linguistic_aux = []
        for _, _, conds_linguistic_names in os.walk(
                speaker_conds_linguistic_dir
        ):
            for conds_linguistic_name in conds_linguistic_names:
                if conds_linguistic_name[-4:] != '.lab':
                    continue

                utterance_lab_path = os.path.join(
                    speaker_conds_linguistic_dir, conds_linguistic_name
                )
                for lab_line in SampleRNNPASEDataset.read_lab(
                        utterance_lab_path
                ):
                    # PHONEMES Features: p1, p2, p3, p4, p5
                    for i in operator.itemgetter(
                            *self.PHONEME_FEATURES_INDEXES
                    )(lab_line):
                        self.categories['phonemes'].add(i)

                    # Vowel Features: b16
                    self.categories['vowels'].add(
                        lab_line[self.VOWEL_FEATURES_INDEXES[0]]
                    )

                    # GPOS Features: d1, e1, f1
                    for i in operator.itemgetter(
                            *self.GPOS_FEATURES_INDEXES
                    )(lab_line):
                        self.categories['gpos'].add(i)

                    # TOBI Features: h5
                    self.categories['tobi'] \
                        .add(lab_line[self.TOBI_FEATURES_INDEXES[0]])

                    # Store the duration to obtain the MEAN and STD
                    conds_linguistic_aux.append(
                        int(lab_line[1]) - int(lab_line[0])
                    )

                    # Real Features: p6, p7, a3, b3, b4, b5, b6, b7, b8, b9,
                    # b10, b11, b12, b13, b14, b15, c3, d2, e2, e3, e4, e5,
                    # e6, e7, e8, f2, g1, g2, h1, h2, h3, h4, i1, i2, j1, j2,
                    # j3
                    conds_linguistic_aux += list(operator.itemgetter(
                        *self.REAL_FEATURES_INDEXES
                    )(lab_line))

        # Reshape and fix unknown values of the auxiliar variable to perform
        # MEAN and STD
        utterances_aux_np = np.asarray(conds_linguistic_aux).reshape(-1, 38)
        utterances_aux_np[utterances_aux_np == 'x'] = 0
        utterances_aux_np = utterances_aux_np.astype(np.float)

        # Create vector of MEANs and STDs
        utterances_means = np.zeros(55, )
        utterances_stds = np.ones(55, )

        # Assign the values to the correct positions
        np.put(
            utterances_means, [0] + self.REAL_FEATURES_INDEXES,
            np.mean(utterances_aux_np, axis=0)
        )

        # Assign the values to the correct positions
        np.put(
            utterances_stds, [0] + self.REAL_FEATURES_INDEXES,
            np.std(utterances_aux_np, axis=0)
        )

        # Store MEAN and STD in the speaker
        speaker['conds_linguistic_stads'] = (
            utterances_means.tolist(), utterances_stds.tolist()
        )

    def _allocate_speakers_ids(self):
        """Fills the parameter ``self.speakers_ids``.
        """
        male_speakers_ids = [
            speaker_id for speaker_id, speaker in self.info['speakers'].items()
            if speaker['gender'] == 'M'
        ]
        if len(male_speakers_ids) < (
                self.kwargs['modeling_male_speakers'] +
                self.kwargs['adaptation_male_speakers']
        ):
            exit('There are not enough male speakers')

        female_speakers_ids = [
            speaker_id for speaker_id, speaker in self.info['speakers'].items()
            if speaker['gender'] == 'F'
        ]
        if len(female_speakers_ids) < (
                self.kwargs['modeling_female_speakers'] +
                self.kwargs['adaptation_female_speakers']
        ):
            exit('There are not enough female speakers')

        # By default, order the lists by DESC duration of the speakers wavs
        if self.kwargs['priorize_longer_speakers']:
            male_speakers_ids = sorted(
                male_speakers_ids,
                key=(lambda speaker_id: self.info['speakers'][speaker_id][
                    'wavs_len']),
                reverse=True
            )
            female_speakers_ids = sorted(
                female_speakers_ids,
                key=(lambda speaker_id: self.info['speakers'][speaker_id][
                    'wavs_len']),
                reverse=True
            )
        else:
            random.shuffle(male_speakers_ids)
            random.shuffle(female_speakers_ids)

        self.speakers_ids['modeling'] = \
            male_speakers_ids[:self.kwargs['modeling_male_speakers']] + \
            female_speakers_ids[:self.kwargs['modeling_female_speakers']]
        self.speakers_ids['adaptation'] = \
            male_speakers_ids[-self.kwargs['adaptation_male_speakers']:] + \
            female_speakers_ids[-self.kwargs['adaptation_female_speakers']:]
        self.speakers_ids['modeling'].sort()
        self.speakers_ids['adaptation'].sort()

    def _allocate_utterances_ids(self):
        """Fills the parameter ``self.utterances_ids``.
        """
        for x in ['modeling', 'adaptation']:
            speakers_ids = self.speakers_ids[x]

            for speaker_id in speakers_ids:
                utterances_ids = [
                    utterance_id for utterance_id, utterance in
                    self.info['utterances'].items()
                    if utterance['speaker_id'] == speaker_id
                ]
                random.shuffle(utterances_ids)

                for y in ['test', 'val', 'train']:
                    time_acc, max_time = 0, getattr(
                        self, '{}_{}_time_per_speaker'.format(x, y)
                    )
                    utterances_list = self.utterances_ids[x][y]

                    while time_acc < max_time:
                        if len(utterances_ids) > 0:
                            utterance_id = utterances_ids.pop()
                            utterances_list.append(utterance_id)
                            time_acc += self.info['utterances'][utterance_id][
                                'wav_len']
                        else:
                            break

                    utterances_list.sort()

    def train_dataloader(self):
        """Returns the data loader containing the training data.

        Returns:
            Data loader containing the training data.
        """
        train_dataset = SampleRNNPASEDataset(
            self.info, self.speakers_ids['modeling'],
            self.utterances_ids['modeling']['train'], self.categories,
            **self.kwargs
        )
        return SampleRNNPASELoader(train_dataset, self.kwargs['batch_size'])

    def val_dataloader(self):
        """Returns the data loader containing the validation data.

        Returns:
            Data loader containing the validation data.
        """
        val_dataset = SampleRNNPASEDataset(
            self.info, self.speakers_ids['modeling'],
            self.utterances_ids['modeling']['val'], self.categories,
            **self.kwargs
        )
        return SampleRNNPASELoader(val_dataset, self.kwargs['batch_size'])

    def test_dataloader(self):
        """Returns the data loader containing the test data.

        Returns:
            Data loader containing the test data.
        """
        test_dataset = SampleRNNPASEDataset(
            self.info, self.speakers_ids['modeling'],
            self.utterances_ids['modeling']['test'], self.categories,
            **self.kwargs
        )
        test_samples_ids = random.sample(
            list(range(len(self.utterances_ids['modeling']['test']))), 10
        )
        return torch.utils.data.DataLoader(
            test_dataset, 1,
            sampler=torch.utils.data.SubsetRandomSampler(test_samples_ids)
        )

    def get_conds_linguistic_size(self):
        """Returns the length of the different categorical linguistic features.

        Returns:
            A list of four items containing the number of different phonemes,
            vowels, GPOS and TOBI categorical features.
        """
        return [
            len(self.categories['phonemes']), len(self.categories['vowels']),
            len(self.categories['gpos']), len(self.categories['tobi'])
        ]

    @staticmethod
    def add_data_specific_args(parent_parser):
        """Adds data-related CLI arguments to the parser.

        Args:
            parent_parser: Parser object just before adding the arguments.

        Returns:
            Parser object after adding the arguments.
        """
        parser = parent_parser.add_argument_group('SampleRNNPASEData')
        parser.add_argument(
            '--data_path', default='./data/'
        )
        parser.add_argument(
            '--data_ckpt_path', default='./lightning_logs/data.ckpt'
        )
        parser.add_argument('--modeling_male_speakers', type=int, default=20)
        parser.add_argument('--modeling_female_speakers', type=int, default=20)
        parser.add_argument(
            '--modeling_train_time_per_speaker', type=int, default=9999
        )
        parser.add_argument(
            '--modeling_val_time_per_speaker', type=int, default=45
        )
        parser.add_argument(
            '--modeling_test_time_per_speaker', type=int, default=45
        )
        parser.add_argument('--adaptation_male_speakers', type=int, default=5)
        parser.add_argument(
            '--adaptation_female_speakers', type=int, default=5
        )
        parser.add_argument(
            '--adaptation_train_time_per_speaker', type=int, default=9999
        )
        parser.add_argument(
            '--adaptation_val_time_per_speaker', type=int, default=30
        )
        parser.add_argument(
            '--adaptation_test_time_per_speaker', type=int, default=30
        )
        parser.add_argument(
            '--priorize_longer_speakers', type=bool, default=True
        )
        parser.add_argument('--batch_size', type=int, default=64)
        return parent_parser
