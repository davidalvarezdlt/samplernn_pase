import random
import torch
import torch.utils.data


class SampleRNNPASELoader(torch.utils.data.DataLoader):
    dataset = None
    batch_size = None
    receptive_field = None
    conds_utterance_size = None

    dataset_iterator = None
    buffer = None
    reset_buffer = None
    no_more_samples_in_batch = None

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.receptive_field = self.dataset.frame_size * self.dataset.sequence_length
        if self.dataset.conds_utterance_type == 'acoustic':
            self.conds_utterance_size = 43
        elif self.dataset.conds_utterance_type == 'linguistic':
            self.conds_utterance_size = 55
        else:
            self.conds_utterance_size = 57
        super().__init__(self.dataset, self.batch_size)

    def __iter__(self):
        self._reset_parameters()
        while True:
            self._prepare_buffers()
            self._fill_data()
            yield self._yield_iteration()

    def _reset_parameters(self):
        self.dataset.shuffle_utterances()
        self.dataset_iterator = iter(self.dataset)
        self.buffer = [None] * self.batch_size
        self.reset_buffer = [None] * self.batch_size
        self.no_more_samples_in_batch = False

    def _prepare_buffers(self):
        for buffer_index, buffer_item in enumerate(self.buffer):
            if buffer_item is None:
                continue
            self.reset_buffer[buffer_index] = False
            if buffer_item[1].shape[0] < self.dataset.sequence_length:
                self.buffer[buffer_index] = None
                self.reset_buffer[buffer_index] = None

    def _fill_data(self):
        while not all(self.buffer) and not self.no_more_samples_in_batch:
            try:
                none_indexes = [i for i, x in enumerate(self.buffer) if x is None]
                none_index = random.choice(none_indexes)
                self.buffer[none_index] = list(next(self.dataset_iterator))
                self.reset_buffer[none_index] = True
            except StopIteration:
                self.no_more_samples_in_batch = True

    def _yield_iteration(self):
        x_len, y_len, utt_conds_len = self._get_iteration_sizes()
        x = []
        y = []
        utt_conds = []
        reset = [2 if reset_item is None else int(reset_item) for i, reset_item in enumerate(self.reset_buffer)]
        info = [buffer_item[2] if buffer_item is not None else None for i, buffer_item in enumerate(self.buffer)]
        for buffer_index, buffer_item in enumerate(self.buffer):
            if buffer_item is None:
                x.append(torch.zeros(x_len))
                y.append(torch.zeros(y_len))
                utt_conds.append(torch.zeros(utt_conds_len, self.conds_utterance_size))
                continue
            else:
                x.append(torch.from_numpy(buffer_item[0][:x_len]))
                y.append(torch.from_numpy(buffer_item[0][self.dataset.frame_size:self.dataset.frame_size + y_len]))
                utt_conds.append(torch.from_numpy(buffer_item[1][:utt_conds_len, :]).type(torch.float32))
                buffer_item[0] = buffer_item[0][y_len:]
                buffer_item[1] = buffer_item[1][utt_conds_len:, :]
        return torch.stack(x), torch.stack(y), torch.stack(utt_conds), torch.tensor(reset), info

    def _get_iteration_sizes(self):
        return self.receptive_field + self.dataset.frame_size - 1, self.receptive_field, self.dataset.sequence_length
