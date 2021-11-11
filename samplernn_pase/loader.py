"""
Module containing the ``torch.utils.data.DataLoader`` implementation of the
package.
"""
import random

import torch
import torch.utils.data


class SampleRNNPASELoader(torch.utils.data.DataLoader):
    """Implementation of the ``torch.utils.data.DataLoader`` used in this
    package.

    Due to the recurrent nature of the model, we need to keep some
    consistency between batch iterations. To do so, the same utterance
    should be allocated to the same batch position until it has been
    completely consumed.

    Attributes:
        dataset: Instance of a ``SampleRNNPASEDataset`` object.
        batch_size: ``--batch_size`` CLI parameter.
        receptive_field: Number of samples that the model sees at every
            iteration.
        conds_utterance_size: Size of the conditionings features vector.
        dataset_iterator: Iterator of ``self.dataset``.
        buffer: List of of length ``self.batch_size`` containing the
            utterances that are currently being consumed.
        reset_buffer: List of length ``self.batch_size`` indicating if the
            sample at that position has just been inserted or not.
        no_more_samples_in_batch: Boolean indicating if the dataset does not
            have more data to consume.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.receptive_field = \
            dataset.kwargs['sequence_length'] * dataset.frame_size
        if dataset.kwargs['conds_utterance_type'] == 'acoustic':
            self.conds_utterance_size = 43
        elif dataset.kwargs['conds_utterance_type'] == 'linguistic':
            self.conds_utterance_size = 55
        else:
            self.conds_utterance_size = 57
        super().__init__(self.dataset, self.batch_size)

    def __iter__(self):
        """Yields an iteration of the data.

        Yields:
            Tuple of five positions containing:
                - A tensor containing the samples used as input.
                - A tensor containing the samples to be predicted.
                - A tensor containing the conditionings.
                - A tensor of the same size as the batch containing 1 if
                    the portion of that sample is the start of the utterance.
                - The metadata of the items, including utterance, speaker
                    and dataset metadata.
        """
        self._reset_parameters()
        while True:
            self._prepare_buffers()
            self._fill_data()
            yield self._yield_iteration()

    def _reset_parameters(self):
        """Resets the parameters of the loader.
        """
        self.dataset.shuffle_utterances()
        self.dataset_iterator = iter(self.dataset)
        self.buffer = [None] * self.batch_size
        self.reset_buffer = [None] * self.batch_size
        self.no_more_samples_in_batch = False

    def _prepare_buffers(self):
        """Prepares the buffers before they are consumed.

        Adjusts the state of ``self.reset_buffer`` and removes those buffer
        positions which do not have enough data.
        """
        for buffer_index, buffer_item in enumerate(self.buffer):
            if buffer_item is None:
                continue
            self.reset_buffer[buffer_index] = False
            if buffer_item[1].shape[0] < \
                    self.dataset.kwargs['sequence_length']:
                self.buffer[buffer_index] = None
                self.reset_buffer[buffer_index] = None

    def _fill_data(self):
        """Fills the positions of the buffer which have already completely
        consumed.

        If there is no more data to fill the buffers with, it sets
        ``self.no_more_samples_in_batch`` to ``True``.
        """
        while not all(self.buffer) and not self.no_more_samples_in_batch:
            try:
                none_indexes = [
                    i for i, x in enumerate(self.buffer) if x is None
                ]
                none_index = random.choice(none_indexes)
                self.buffer[none_index] = list(next(self.dataset_iterator))
                self.reset_buffer[none_index] = True
            except StopIteration:
                self.no_more_samples_in_batch = True

    def _yield_iteration(self):
        """Returns the data associated with one iteration.

        The method reads from the buffers and removes the data that is about
        to be returned.

        Returns:
            Tuple of five positions containing:
                - A tensor containing the samples used as input.
                - A tensor containing the samples to be predicted.
                - A tensor containing the conditionings.
                - A tensor of the same size as the batch containing 1 if
                    the portion of that sample is the start of the utterance.
                - The metadata of the items, including utterance, speaker
                    and dataset metadata.
        """
        x_len, y_len, utt_conds_len = self._get_iteration_sizes()
        x, y, utt_conds = [], [], []
        reset = [
            2 if reset_item is None else int(reset_item)
            for i, reset_item in enumerate(self.reset_buffer)
        ]
        info = [
            buffer_item[2] if buffer_item is not None else None
            for i, buffer_item in enumerate(self.buffer)
        ]
        for buffer_index, buffer_item in enumerate(self.buffer):
            if buffer_item is None:
                x.append(torch.zeros(x_len))
                y.append(torch.zeros(y_len))
                utt_conds.append(
                    torch.zeros(utt_conds_len, self.conds_utterance_size)
                )
                continue
            else:
                x.append(torch.from_numpy(buffer_item[0][:x_len]))
                y.append(torch.from_numpy(buffer_item[0][
                    self.dataset.frame_size:self.dataset.frame_size + y_len]
                ))
                utt_conds.append(
                    torch.from_numpy(buffer_item[1][:utt_conds_len, :])
                )
                buffer_item[0] = buffer_item[0][y_len:]
                buffer_item[1] = buffer_item[1][utt_conds_len:, :]
        return torch.stack(x), torch.stack(y), torch.stack(utt_conds), \
            torch.tensor(reset), info

    def _get_iteration_sizes(self):
        """Returns the length of the input samples, the lenght of the
        samples to be predicted and the lenght of the conditionings.

        Returns:
            Tuple of three positions containing:
                - The length of the input samples.
                - The length of the samples to be predicted.
                - The lenght of the conditionings.
        """
        return self.receptive_field + self.dataset.frame_size - 1, \
            self.receptive_field, self.dataset.kwargs['sequence_length']
