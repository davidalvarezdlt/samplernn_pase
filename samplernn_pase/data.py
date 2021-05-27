import ahoproc_tools.io
import ahoproc_tools.interpolate
import concurrent.futures
import csv
import numpy as np
import os.path
import operator
import progressbar
import random
import samplernn_pase.dataset
import samplernn_pase.loader
import samplernn_pase.utils as utils
import skeltorch
import soundfile
import torch.utils.data


class SampleRNNPASEData(skeltorch.Data):
    datasets_info = {}
    speakers_info = {}
    utterances_info = {}
    utterances_conds_linguistic_categories = {
        'phonemes': set(),
        'vowels': set(),
        'gpos': set(),
        'tobi': set()
    }
    modeling_speakers_ids = []
    modeling_utterances_ids_train = []
    modeling_utterances_ids_val = []
    modeling_utterances_ids_test = []
    modeling_utterances_ids_test_infer = []
    adaptation_speakers_ids = []
    adaptation_utterances_ids_train = []
    adaptation_utterances_ids_val = []
    adaptation_utterances_ids_test = []

    def create(self, data_path):
        self._load_datasets_info(data_path)
        for dataset_id, _ in self.datasets_info.items():
            self._load_speakers_info(dataset_id)
        bar = progressbar.ProgressBar(max_value=len(self.speakers_info.items()), initial_value=1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for speaker_id, _ in self.speakers_info.items():
                executor.submit(self._load_speaker_data, speaker_id, bar)
        self._allocate_speakers_ids()
        self._allocate_utterances_ids()

    def _load_datasets_info(self, data_path):
        self.datasets_info['vctk'] = {
            'index': len(self.datasets_info),
            'name': 'vctk',
            'speakers_prefix': 'vctk_',
            'info_file_path': os.path.join(data_path, 'vctk', 'speaker-info.txt'),
            'wavs_folder_path': os.path.join(data_path, 'vctk', 'wav16_trimmed'),
            'conds_utterance': {
                'acoustic_folder_path': os.path.join(data_path, 'vctk', 'wav16_trimmed_conds_acoustic'),
                'linguistic_folder_path': os.path.join(data_path, 'vctk', 'wav16_trimmed_conds_linguistic'),
            },
            'speakers_count': 108,
            'utterances_sf': 16000
        }
        self.datasets_info['cmu_arctic'] = {
            'index': len(self.datasets_info),
            'name': 'cmu_artic',
            'speakers_prefix': 'cmu_',
            'info_file_path': os.path.join(data_path, 'cmu_arctic', 'speaker-info.txt'),
            'wavs_folder_path': os.path.join(data_path, 'cmu_arctic', 'wav16_trimmed'),
            'conds_utterance': {
                'acoustic_folder_path': os.path.join(data_path, 'cmu_arctic', 'wav16_trimmed_conds_acoustic'),
                'linguistic_folder_path': os.path.join(data_path, 'cmu_arctic', 'wav16_trimmed_conds_linguistic'),
            },
            'speakers_count': 18,
            'utterances_sf': 16000
        }

    def _load_speakers_info(self, dataset_id):
        dataset = self.datasets_info[dataset_id]
        info_list = self._read_dataset_info_file_vctk() if dataset_id == 'vctk' else self._read_dataset_info_file_cmu()
        for speaker_name, speaker_gender in info_list:
            self.speakers_info[dataset['speakers_prefix'] + speaker_name] = {
                'index': len(self.speakers_info),
                'dataset_id': dataset_id,
                'name': speaker_name,
                'gender': speaker_gender,
                'wavs_len': 0,
                'conds_acoustic_stads': (0, 0),
                'conds_linguistic_stads': (0, 0)
            }

    def _read_dataset_info_file_vctk(self):
        info_list = []
        with open(self.datasets_info['vctk']['info_file_path'], 'r') as info_file:
            speakers_info = csv.reader(info_file, delimiter=' ')
            speakers_info_it = iter(speakers_info)
            next(speakers_info_it)
            for speakers_info_row in speakers_info_it:
                info_list.append(('p' + speakers_info_row[0], speakers_info_row[4]))
        return info_list

    def _read_dataset_info_file_cmu(self):
        info_list = []
        with open(self.datasets_info['cmu_arctic']['info_file_path'], 'r') as info_file:
            speakers_info = csv.reader(info_file, delimiter=',')
            speakers_info_it = iter(speakers_info)
            next(speakers_info_it)
            for speakers_info_row in speakers_info_it:
                info_list.append((speakers_info_row[0], speakers_info_row[1]))
        return info_list

    def _load_speaker_data(self, speaker_id, bar):
        self._load_utterances_info(speaker_id)
        self._load_conds_acoustic_stads(speaker_id)
        self._load_conds_linguistic_stads(speaker_id)
        bar.update(max(bar.value, bar.value + 1))

    def _load_utterances_info(self, speaker_id):
        speaker = self.speakers_info[speaker_id]
        dataset = self.datasets_info[speaker['dataset_id']]
        speakers_wavs_folder_path = os.path.join(dataset['wavs_folder_path'], speaker['name'])
        for _, _, utterances_names in os.walk(speakers_wavs_folder_path):
            for utterance_name in utterances_names:
                if utterance_name[-4:] != '.wav':
                    continue
                utterance_id = speaker_id + '-' + utterance_name[:-4]
                utterance, _ = soundfile.read(os.path.join(speakers_wavs_folder_path, utterance_name))
                self.utterances_info[utterance_id] = {
                    'index': len(self.utterances_info),
                    'speaker_id': speaker_id,
                    'name': utterance_name[:-4],
                    'wav_len': utterance.size / dataset['utterances_sf'],
                    'path': speaker['name'] + os.sep + utterance_name[:-4]
                }
                speaker['wavs_len'] += self.utterances_info[utterance_id]['wav_len']

    def _load_conds_acoustic_stads(self, speaker_id):
        speaker = self.speakers_info[speaker_id]
        dataset = self.datasets_info[speaker['dataset_id']]
        speaker_conds_acoustic_dir = os.path.join(dataset['conds_utterance']['acoustic_folder_path'], speaker['name'])
        utterances_aux = None
        for _, _, utterances_names in os.walk(speaker_conds_acoustic_dir):
            for utterance_name in utterances_names:
                if utterance_name[-3:] != '.cc':
                    continue
                utterance_cc = ahoproc_tools.io.read_aco_file(
                    os.path.join(speaker_conds_acoustic_dir, utterance_name), (-1, 40)
                )
                utterance_fv = ahoproc_tools.io.read_aco_file(
                    os.path.join(speaker_conds_acoustic_dir, utterance_name[:-3] + '.fv'), (-1)
                )
                utterance_lf0 = ahoproc_tools.io.read_aco_file(
                    os.path.join(speaker_conds_acoustic_dir, utterance_name[:-3] + '.lf0'), (-1)
                )

                # Interpolate both FV and LF0 and obtain VU
                utterance_fv, _ = ahoproc_tools.interpolate.interpolation(utterance_fv, 1e3)
                utterance_lf0, utterance_vu = ahoproc_tools.interpolate.interpolation(utterance_lf0, -1e10)

                # Compute LOG(FV)
                utterance_fv = np.log(utterance_fv)

                # Merge all the conds, set 1 in the last position so we get MEAN=0 & STD=0 for VU bool conditionant
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
                    utterances_aux = np.concatenate([utterances_aux, utterance_conds])

        # Compute the MEAN and STD
        utterances_means = np.mean(utterances_aux, axis=0)
        utterances_stds = np.std(utterances_aux, axis=0)

        # Set the STD=1 for the bool conditionant
        utterances_stds[-1] = 1

        # Store MEAN and STD in the information of the speaker
        speaker['conds_acoustic_stads'] = (utterances_means.tolist(), utterances_stds.tolist())

    def _load_conds_linguistic_stads(self, speaker_id):
        speaker = self.speakers_info[speaker_id]
        dataset = self.datasets_info[speaker['dataset_id']]
        speaker_conds_linguistic_dir = os.path.join(
            dataset['conds_utterance']['linguistic_folder_path'], speaker['name']
        )
        conds_linguistic_aux = []
        for _, _, conds_linguistic_names in os.walk(speaker_conds_linguistic_dir):
            for conds_linguistic_name in conds_linguistic_names:
                if conds_linguistic_name[-4:] != '.lab':
                    continue
                for lab_line in utils.read_lab(os.path.join(speaker_conds_linguistic_dir, conds_linguistic_name)):

                    # PHONEMES Features: p1, p2, p3, p4, p5
                    for i in operator.itemgetter(2, 3, 4, 5, 6)(lab_line):
                        self.utterances_conds_linguistic_categories['phonemes'].add(i)

                    # Vowel Features: b16
                    self.utterances_conds_linguistic_categories['vowels'].add(lab_line[27])

                    # GPOS Features: d1, e1, f1
                    for i in operator.itemgetter(31, 33, 41)(lab_line):
                        self.utterances_conds_linguistic_categories['gpos'].add(i)

                    # TOBI Features: h5
                    self.utterances_conds_linguistic_categories['tobi'].add(lab_line[49])

                    # Store the duration to obtain the MEAN and STD
                    conds_linguistic_aux.append(int(lab_line[1]) - int(lab_line[0]))

                    # Real Features: p6, p7, a3, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, c3, d2, e2,
                    # e3, e4, e5, e6, e7, e8, f2, g1, g2, h1, h2, h3, h4, i1, i2, j1, j2, j3
                    conds_linguistic_aux += list(operator.itemgetter(
                        7, 8, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30, 32, 34, 35, 36, 37, 38, 39,
                        40,
                        42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54
                    )(lab_line))

        # Reshape and fix unknown values of the auxiliar variable to perform MEAN and STD
        utterances_aux_np = np.asarray(conds_linguistic_aux).reshape(-1, 38)
        utterances_aux_np[utterances_aux_np == 'x'] = 0
        utterances_aux_np = utterances_aux_np.astype(np.float)

        # Create vector of MEANs and STDs
        utterances_means = np.zeros(55, )
        utterances_stds = np.ones(55, )

        # Assign the values to the correct positions
        np.put(
            a=utterances_means,
            ind=[0, 7, 8, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30, 32, 34, 35, 36, 37, 38, 39, 40,
                 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54],
            v=np.mean(utterances_aux_np, axis=0)
        )

        # Assign the values to the correct positions
        np.put(
            a=utterances_stds,
            ind=[0, 7, 8, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30, 32, 34, 35, 36, 37, 38, 39, 40,
                 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54],
            v=np.std(utterances_aux_np, axis=0)
        )

        # Store MEAN and STD in the speaker
        speaker['conds_linguistic_stads'] = (utterances_means.tolist(), utterances_stds.tolist())

    def _allocate_speakers_ids(self):
        male_speakers_ids = [
            speaker_id for speaker_id, speaker in self.speakers_info.items() if speaker['gender'] == 'M'
        ]
        female_speakers_ids = [
            speaker_id for speaker_id, speaker in self.speakers_info.items() if speaker['gender'] == 'F'
        ]
        modeling_male_speakers = self.experiment.configuration.get('data', 'modeling_male_speakers')
        modeling_female_speakers = self.experiment.configuration.get('data', 'modeling_female_speakers')
        adaptation_male_speakers = self.experiment.configuration.get('data', 'adaptation_male_speakers')
        adaptation_female_speakers = self.experiment.configuration.get('data', 'adaptation_female_speakers')

        # Verify that there are enough male speakers
        if len(male_speakers_ids) < (modeling_male_speakers + adaptation_male_speakers):
            self.logger.error('There are not enough male speakers')
            exit()

        # Verify that there are enough female speakers
        if len(female_speakers_ids) < (modeling_female_speakers + adaptation_female_speakers):
            self.logger.error('There are not enough female speakers')
            exit()

        # By default, order the lists by DESC duration of the speakers wavs
        if self.experiment.configuration.get('data', 'priorize_longer_speakers'):
            male_speakers_ids = sorted(
                male_speakers_ids, key=(lambda speaker_id: self.speakers_info[speaker_id]['wavs_len']), reverse=True
            )
            female_speakers_ids = sorted(
                female_speakers_ids, key=(lambda speaker_id: self.speakers_info[speaker_id]['wavs_len']), reverse=True
            )
        else:
            random.shuffle(male_speakers_ids)
            random.shuffle(female_speakers_ids)

        # Assign the speakers to the lists
        self.modeling_speakers_ids = \
            male_speakers_ids[:modeling_male_speakers] + female_speakers_ids[:modeling_female_speakers]
        self.adaptation_speakers_ids = \
            male_speakers_ids[-adaptation_male_speakers:] + female_speakers_ids[-adaptation_female_speakers:]
        self.modeling_speakers_ids.sort()
        self.adaptation_speakers_ids.sort()

    def _allocate_utterances_ids(self, shuffle=True):
        for modeling_speaker_id in self.modeling_speakers_ids:
            modeling_speaker_utterances_ids = [
                utterance_id for utterance_id, utterance in self.utterances_info.items()
                if utterance['speaker_id'] == modeling_speaker_id
            ]
            if shuffle:
                random.shuffle(modeling_speaker_utterances_ids)
            modeling_speaker_test_time_acc = 0
            modeling_speaker_val_time_acc = 0
            modeling_speaker_train_time_acc = 0
            while modeling_speaker_test_time_acc < \
                    self.experiment.configuration.get('data', 'modeling_test_time_per_speaker'):
                if len(modeling_speaker_utterances_ids) > 0:
                    modeling_speaker_utterance_id = modeling_speaker_utterances_ids.pop()
                    self.modeling_utterances_ids_test.append(modeling_speaker_utterance_id)
                    modeling_speaker_test_time_acc += self.utterances_info[modeling_speaker_utterance_id]['wav_len']
                else:
                    self.logger.warning('Not enough data for {}'.format(modeling_speaker_id))
                    break
            self.modeling_utterances_ids_test_infer = random.sample(
                list(range(len(self.modeling_utterances_ids_test))), 10
            )
            while modeling_speaker_val_time_acc < \
                    self.experiment.configuration.get('data', 'modeling_val_time_per_speaker'):
                if len(modeling_speaker_utterances_ids) > 0:
                    modeling_speaker_utterance_id = modeling_speaker_utterances_ids.pop()
                    self.modeling_utterances_ids_val.append(modeling_speaker_utterance_id)
                    modeling_speaker_val_time_acc += self.utterances_info[modeling_speaker_utterance_id]['wav_len']
                else:
                    self.logger.warning('Not enough data for {}'.format(modeling_speaker_id))
                    break
            while modeling_speaker_train_time_acc < \
                    self.experiment.configuration.get('data', 'modeling_train_time_per_speaker'):
                if len(modeling_speaker_utterances_ids) > 0:
                    modeling_speaker_utterance_id = modeling_speaker_utterances_ids.pop()
                    self.modeling_utterances_ids_train.append(modeling_speaker_utterance_id)
                    modeling_speaker_train_time_acc += \
                        self.utterances_info[modeling_speaker_utterance_id]['wav_len']
                else:
                    self.logger.warning('Not enough data for {}'.format(modeling_speaker_id))
                    break
        for adaptation_speaker_id in self.adaptation_speakers_ids:
            adaptation_speaker_utterances_ids = [
                utterance_id for utterance_id, utterance in self.utterances_info.items()
                if utterance['speaker_id'] == adaptation_speaker_id
            ]
            if shuffle:
                random.shuffle(adaptation_speaker_utterances_ids)
            adaptation_speaker_test_time_acc = 0
            adaptation_speaker_val_time_acc = 0
            adaptation_speaker_train_time_acc = 0
            while adaptation_speaker_test_time_acc < \
                    self.experiment.configuration.get('data', 'adaptation_test_time_per_speaker'):
                if len(adaptation_speaker_utterances_ids) > 0:
                    adaptation_speaker_utterance_id = adaptation_speaker_utterances_ids.pop()
                    self.adaptation_utterances_ids_test.append(adaptation_speaker_utterance_id)
                    adaptation_speaker_test_time_acc += \
                        self.utterances_info[adaptation_speaker_utterance_id]['wav_len']
                else:
                    self.logger.warning('Not enough data for {}'.format(adaptation_speaker_id))
                    break
            while adaptation_speaker_val_time_acc < \
                    self.experiment.configuration.get('data', 'adaptation_val_time_per_speaker'):
                if len(adaptation_speaker_utterances_ids) > 0:
                    adaptation_speaker_utterance_id = adaptation_speaker_utterances_ids.pop()
                    self.adaptation_utterances_ids_val.append(adaptation_speaker_utterance_id)
                    adaptation_speaker_val_time_acc += \
                        self.utterances_info[adaptation_speaker_utterance_id]['wav_len']
                else:
                    self.logger.warning('Not enough data for {}'.format(adaptation_speaker_id))
                    break
            while adaptation_speaker_train_time_acc < \
                    self.experiment.configuration.get('data', 'adaptation_train_time_per_speaker'):
                if len(adaptation_speaker_utterances_ids) > 0:
                    adaptation_speaker_utterance_id = adaptation_speaker_utterances_ids.pop()
                    self.adaptation_utterances_ids_train.append(adaptation_speaker_utterance_id)
                    adaptation_speaker_train_time_acc += \
                        self.utterances_info[adaptation_speaker_utterance_id]['wav_len']
                else:
                    self.logger.warning('Not enough data for {}'.format(adaptation_speaker_id))
                    break
        self.modeling_utterances_ids_train.sort()
        self.modeling_utterances_ids_val.sort()
        self.modeling_utterances_ids_test.sort()
        self.adaptation_utterances_ids_train.sort()
        self.adaptation_utterances_ids_val.sort()
        self.adaptation_utterances_ids_test.sort()

    def load_datasets(self, data_path):
        self.datasets['train'] = samplernn_pase.dataset.SampleRNNPASEDataset(
            datasets_info=self.datasets_info,
            speakers_info=self.speakers_info,
            utterances_info=self.utterances_info,
            speakers_ids=self.modeling_speakers_ids,
            utterances_ids=self.modeling_utterances_ids_train,
            conds_utterance_type=self.experiment.configuration.get('conditionals', 'conds_utterance_type'),
            utterances_conds_linguistic_categories=self.utterances_conds_linguistic_categories,
            sequence_length=self.experiment.configuration.get('model', 'sequence_length'),
            frame_layers_ratios=self.experiment.configuration.get('model', 'ratios'),
            split='train'
        )
        self.datasets['validation'] = samplernn_pase.dataset.SampleRNNPASEDataset(
            datasets_info=self.datasets_info,
            speakers_info=self.speakers_info,
            utterances_info=self.utterances_info,
            speakers_ids=self.modeling_speakers_ids,
            utterances_ids=self.modeling_utterances_ids_val,
            conds_utterance_type=self.experiment.configuration.get('conditionals', 'conds_utterance_type'),
            utterances_conds_linguistic_categories=self.utterances_conds_linguistic_categories,
            sequence_length=self.experiment.configuration.get('model', 'sequence_length'),
            frame_layers_ratios=self.experiment.configuration.get('model', 'ratios'),
            split='validation'
        )
        self.datasets['test'] = samplernn_pase.dataset.SampleRNNPASEDataset(
            datasets_info=self.datasets_info,
            speakers_info=self.speakers_info,
            utterances_info=self.utterances_info,
            speakers_ids=self.modeling_speakers_ids,
            utterances_ids=self.modeling_utterances_ids_test,
            conds_utterance_type=self.experiment.configuration.get('conditionals', 'conds_utterance_type'),
            utterances_conds_linguistic_categories=self.utterances_conds_linguistic_categories,
            sequence_length=self.experiment.configuration.get('model', 'sequence_length'),
            frame_layers_ratios=self.experiment.configuration.get('model', 'ratios'),
            split='test'
        )

    def load_loaders(self, data_path, num_workers):
        self.loaders['train'] = samplernn_pase.loader.SampleRNNPASELoader(
            dataset=self.datasets['train'], batch_size=self.experiment.configuration.get('training', 'batch_size')
        )
        self.loaders['validation'] = samplernn_pase.loader.SampleRNNPASELoader(
            dataset=self.datasets['validation'], batch_size=self.experiment.configuration.get('training', 'batch_size')
        )
        self.loaders['test'] = torch.utils.data.DataLoader(
            dataset=self.datasets['test'], batch_size=1,
            sampler=torch.utils.data.SubsetRandomSampler(self.modeling_utterances_ids_test_infer)
        )

    def get_conds_linguistic_size(self):
        return [
            len(self.utterances_conds_linguistic_categories['phonemes']),
            len(self.utterances_conds_linguistic_categories['vowels']),
            len(self.utterances_conds_linguistic_categories['gpos']),
            len(self.utterances_conds_linguistic_categories['tobi'])
        ]
