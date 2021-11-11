"""
Module containing the ``pytorch_lightning.LightningModule`` implementation of
the package.
"""
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn
import torch.nn.functional as F
import torch.nn.init
import torch.optim
import torch.optim.lr_scheduler


class SampleRNNModel(pl.LightningModule):
    """Implementation of the ``pytorch_lightning.LightningModule`` used in this
    package, representing the SampleRNN network architecture.

    Attributes:
        conds_mixer: Instance of a ``CondsMixer`` layer.
        frames_layers: Instance of a ``torch.nn.ModuleList`` containing
            instances of ``FrameLevelLayer`` layers.
        sample_layer: Instance of a ``SampleLevelLayer`` layer.
        frame_size: Number of samples used as input.
        receptive_field: Number of samples that the model sees at every
            iteration
        kwargs: Dictionary containing the CLI arguments of the execution.
    """

    def __init__(self, n_speakers, ling_features_size, **kwargs):
        super(SampleRNNModel, self).__init__()
        self.conds_mixer = CondsMixer(n_speakers, ling_features_size, **kwargs)
        self.frames_layers = torch.nn.ModuleList()
        frame_layers_fs = list(map(int, np.cumprod(kwargs['ratios'])))
        for layer_n in range(0, len(frame_layers_fs)):
            self.frames_layers.append(
                FrameLevelLayer(layer_n, **kwargs)
            )
        self.sample_layer = SampleLevelLayer(**kwargs)

        self.frame_size = np.prod(kwargs['ratios'])
        self.receptive_field = \
            np.prod(kwargs['ratios']) * kwargs['sequence_length']
        self.kwargs = kwargs

    def _init_rnn_states(self, batch_size):
        """Initializes the class parameter ``self.rnn_states``, which contains
        the last hidden states of the different RNN layers.

        Args:
            batch_size: ``--batch_size`` CLI parameter.
        """
        self.rnn_states = {
            rnn: [None] * batch_size for rnn in self.frames_layers
        }

    def _get_rnn_states(self, layer, reset):
        """Returns the hidden states of the ``layer`` layer for all positions
        of the batch for those positions where ``reset`` is 0. If not, it
        returns None at that batch position.

        Args:
            layer: Instance of a ``FrameLevelLayer`` layer.
            reset: Tensor of size ``self.batch_size`` containing whether the
                sample at the different batch positions are new or recurrent.

        Returns:
            List of RNN states for all elements in the batch.
        """
        return [
            self.rnn_states[layer][reset_index] if reset_element == 0 else None
            for reset_index, reset_element in enumerate(reset)
        ]

    def _set_rnn_states(self, layer, rnn_state, reset):
        """Sets the new RNN state for the layer ``layer`` for all batch
        positions.

        Args:
            layer: Instance of a ``FrameLevelLayer`` layer
            rnn_state: RNN state to be saved.
            reset: Tensor of size ``self.batch_size`` containing whether the
                sample at the different batch positions are new or recurrent.
        """
        for reset_index, reset_element in enumerate(reset):
            if reset_element == 0 or reset_element == 1:
                self.rnn_states[layer][reset_index] = \
                    rnn_state[:, reset_index, :]
            else:
                self.rnn_states[layer][reset_index] = None

    def _quantize(self, x, mu=torch.tensor(255)):
        """Quantizes a waveform using 8 bits after applying the uLaw.

        Args:
            x: Waveformed encoded between [-1, 1].
            mu: Tensor containing the mu value of the uLaw.

        Returns:
            Waveform encoded between [0, 255] using uLaw.
        """
        y = x.sign() * (1 + mu * x.abs()).log() / (1 + mu).log()
        y = 0.5 * (y + 1)
        y *= (self.kwargs['q_levels'] - 1e-6)
        return y.long()

    def _quantize_zero(self):
        """Quantizes the 0.

        Returns:
            Encoded value of 0.
        """
        return self.kwargs['q_levels'] // 2

    def _dequantize(self, y, mu=torch.tensor(255)):
        """Dequantized a waveform that has been encoded using uLaw.

        Args:
            y: Waveform encoded between [0, 255] using uLaw .
            mu: Tensor containing the mu value of the uLaw.

        Returns:
            Waveformed encoded between [-1, 1].
        """
        y = y.float() * 2 / self.kwargs['q_levels'] - 1
        x = (y.abs() * (1 + mu).log()).exp() - 1
        x = x.sign() * x / mu
        return x

    def forward(self, x, utt_conds, info, reset):
        """Propagates the data through SampleRNN.

        Args:
            x: Tensor containing the samples used as input.
            utt_conds: Tensor containing the content conditionings.
            info: Metadata of the items, including utterance, speaker
                and dataset metadata.
            reset: Tensor of the same size as the batch containing 1 if
                the portion of that sample is the start of the utterance.

        Returns:
            Tensor containing the predicted samples.
        """
        b, t, _ = utt_conds.size()
        if not hasattr(self, 'rnnstates'):
            self._init_rnn_states(b)

        conds = self.conds_mixer(utt_conds, info)

        upper_tier_conditioning = None
        for layer in reversed(self.frames_layers):
            from_index = \
                self.frames_layers[-1].input_samples - layer.input_samples
            to_index = -layer.input_samples + 1

            input_samples = x[:, from_index: to_index]
            input_samples = input_samples.contiguous().view(
                x.size(0), -1, layer.input_samples
            )
            rnn_states = self._get_rnn_states(layer, reset)
            upper_tier_conditioning, rnn_states_new = layer(
                input_samples, conds, upper_tier_conditioning, rnn_states
            )

            self._set_rnn_states(layer, rnn_states_new.detach(), reset)

        from_index = self.frames_layers[-1].input_samples - \
            self.sample_layer.input_samples
        input_samples = self._quantize(x)[:, from_index:]
        y_hat_logits = self.sample_layer(
            input_samples, conds, upper_tier_conditioning
        )

        return F.log_softmax(y_hat_logits, dim=2)

    def infer(self, utt_conds, info):
        """Infers a waveform using its conditionings vectors.

        Args:
            utt_conds: Tensor containing the content conditionings.
            info: Metadata of the items, including utterance, speaker
                and dataset metadata.

        Returns:
            Tensor containing the predicted waveform.
        """
        b, t, _ = utt_conds.size()

        self._init_rnn_states(1)
        y_hat = torch.zeros(
            1, (t + 1) * self.frame_size, dtype=torch.int64
        ).fill_(self._quantize_zero()).to(utt_conds.device)
        frame_level_outputs = [None for _ in self.frames_layers]

        conds = self.conds_mixer(utt_conds, [info])

        for xi in range(self.frame_size, y_hat.shape[1]):
            conds_indx, _ = divmod(xi, self.frame_size)
            conds_indx -= 1

            for layer_index, layer in \
                    reversed(list(enumerate(self.frames_layers))):
                if xi % layer.input_samples != 0:
                    continue

                input_samples = self._dequantize(
                    y_hat[:, xi - layer.input_samples:xi].unsqueeze(1)
                ).to(utt_conds.device)

                if layer_index == len(self.frames_layers) - 1:
                    upper_tier_conditioning = None
                else:
                    frame_index = (xi // layer.input_samples) % \
                                  self.frames_layers[layer_index + 1].ratio
                    upper_tier_conditioning = \
                        frame_level_outputs[layer_index + 1
                                            ][:, frame_index, :].unsqueeze(1)

                frame_level_outputs[layer_index], rnn_states_new = layer(
                    input_samples,
                    conds[:, conds_indx, :].unsqueeze(1),
                    upper_tier_conditioning,
                    self._get_rnn_states(
                        layer, [1 if xi == self.frame_size else 0]
                    )
                )

                self._set_rnn_states(layer, rnn_states_new.detach(), [0])

            input_samples = y_hat[:, xi - self.sample_layer.input_samples:xi] \
                .to(utt_conds.device)
            upper_tier_conditioning = frame_level_outputs[0][
                                      :, xi % self.sample_layer.input_samples,
                                      :].unsqueeze(1)

            y_hat[:, xi] = F.log_softmax(self.sample_layer(
                input_samples,
                conds[:, conds_indx, :].unsqueeze(1),
                upper_tier_conditioning
            ), dim=2).squeeze(1).exp_().multinomial(1).squeeze(1)

        return y_hat

    def training_step(self, batch, batch_idx):
        """Performs a single pass through the training dataset.

        Args:
            batch: Output of a single data loader iteration.
            batch_idx: Index representing the iteration number.

        Returns:
            Computed loss between predictions and ground truths.
        """
        x, y, utt_conds, reset, info = batch
        y_hat = self(x, utt_conds, info, reset)

        loss = F.nll_loss(
            y_hat[reset != 2].view(-1, y_hat.size(2)),
            self._quantize(y)[reset != 2].view(-1)
        )
        self.log('training_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Performs a single pass through the validation dataset.

        Args:
            batch: Output of a single data loader iteration.
            batch_idx: Index representing the iteration number.

        Returns:
            Computed loss between predictions and ground truths.
        """
        x, y, utt_conds, reset, info = batch
        y_hat = self(x, utt_conds, info, reset)

        loss = F.nll_loss(
            y_hat[reset != 2].view(-1, y_hat.size(2)),
            self._quantize(y)[reset != 2].view(-1)
        )
        self.log('validation_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        """Performs a single pass through the test dataset.

        Args:
            batch: Output of a single data loader iteration.
            batch_idx: Index representing the iteration number.

        Returns:
            Computed loss between predictions and ground truths.
        """
        x, utt_conds, info = batch
        y_hat = self.infer(utt_conds, info)
        y_hat = self._dequantize(y_hat)

        self.logger.experiment.add_audio(
            info['utterance']['name'][0], y_hat, self.current_epoch, 16000
        )

    def configure_optimizers(self):
        """Configures the optimizer and LR scheduler used in the package.

        Returns:
            Dictionary containing a configured ``torch.optim.Adam``
            optimizer and ``torch.optim.lr_scheduler.ReduceLROnPlateau``
            scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.kwargs['lr'])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.kwargs['lr_scheduler_patience'],
            factor=self.kwargs['lr_scheduler_factor']
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler,
                'monitor': 'validation_loss'}

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Adds model-related CLI arguments to the parser.

        Args:
            parent_parser: Parser object just before adding the arguments.

        Returns:
            Parser object after adding the arguments.
        """
        parser = parent_parser.add_argument_group('SampleRNNModel')
        parser.add_argument('--q_levels', type=int, default=256)
        parser.add_argument('--sequence_length', type=int, default=13)
        parser.add_argument('--ratios', type=int, nargs='+', default=[20, 4])
        parser.add_argument(
            '--conds_speaker_type',
            choices=['embedding', 'pase'],
            default='embedding'
        )
        parser.add_argument(
            '--conds_utterance_type',
            choices=['acoustic', 'linguistic', 'linguistic_lf0'],
            default='linguistic_lf0'
        )
        parser.add_argument('--conds_speaker_size', type=int, default=15)
        parser.add_argument(
            '--conds_utterance_linguistic_emb_size', type=int, default=15
        )
        parser.add_argument('--conds_size', type=int, default=50)
        parser.add_argument(
            '--rnn_layers', type=int, nargs='+', default=[1, 1]
        )
        parser.add_argument(
            '--rnn_hidden_size', type=int, nargs='+', default=[1024, 1024]
        )
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--lr_scheduler_patience', type=int, default=3)
        parser.add_argument('--lr_scheduler_factor', type=float, default=0.5)
        return parent_parser


class CondsMixer(torch.nn.Module):
    """Implementation of the Conditionings Mixer layer.

    Attributes:
        speaker_embedding: Instance of a ``torch.nn.Embedding`` layer.
        conds_utt_phonemes_emb: Instance of a ``torch.nn.Embedding`` layer.
        conds_utt_vowels_emb: Instance of a ``torch.nn.Embedding`` layer.
        conds_utt_gpos_emb: Instance of a ``torch.nn.Embedding`` layer.
        conds_utt_tobi_emb: Instance of a ``torch.nn.Embedding`` layer.
        conds_utterance_size: Size of the content conditioning embedding.
        conds_utterance_expanded_size: Size of the resulting embedding after
            combining content and speaker embeddings.
        conds_mix: Instance of a ``torch.nn.Linear`` layer.
        kwargs: Dictionary containing the CLI arguments of the execution.
    """

    def __init__(self, n_speakers, ling_features_size, **kwargs):
        super(CondsMixer, self).__init__()
        self.speaker_embedding = torch.nn.Embedding(
            n_speakers, kwargs['conds_speaker_size']
        )
        self.conds_utt_phonemes_emb = torch.nn.Embedding(
            ling_features_size[0],
            kwargs['conds_utterance_linguistic_emb_size']
        )
        self.conds_utt_vowels_emb = torch.nn.Embedding(
            ling_features_size[1],
            kwargs['conds_utterance_linguistic_emb_size']
        )
        self.conds_utt_gpos_emb = torch.nn.Embedding(
            ling_features_size[2],
            kwargs['conds_utterance_linguistic_emb_size']
        )
        self.conds_utt_tobi_emb = torch.nn.Embedding(
            ling_features_size[3],
            kwargs['conds_utterance_linguistic_emb_size']
        )

        if kwargs['conds_utterance_type'] == 'acoustic':
            self.conds_utterance_size = 43
            self.conds_utterance_expanded_size = 43
        elif kwargs['conds_utterance_type'] == 'linguistic':
            self.conds_utterance_size = 55
            self.conds_utterance_expanded_size = \
                55 - 10 + 10 * kwargs['conds_utterance_linguistic_emb_size']
        else:
            self.conds_utterance_size = 57
            self.conds_utterance_expanded_size = \
                57 - 10 + 10 * kwargs['conds_utterance_linguistic_emb_size']

        self.conds_mix = torch.nn.Linear(
            self.conds_utterance_expanded_size + 15, kwargs['conds_size']
        )
        self.kwargs = kwargs

    def forward(self, utt_conds, info):
        """Forward pass through the Conditionings Mixer layer.

        Args:
            utt_conds: Tensor containing the content conditionings.
            info: Metadata of the items, including utterance, speaker
                and dataset metadata.

        Returns:
            Tensor containing the hiddden representation that mixes speaker
            and content conditionings.
        """
        speaker_conds = self._forward_speaker_conds(info, utt_conds.device) \
            .expand(utt_conds.size(0), utt_conds.size(1), -1)
        utt_conds = self._forward_content_conds(utt_conds)
        return self.conds_mix(torch.cat((speaker_conds, utt_conds), dim=2))

    def _forward_speaker_conds(self, info, device):
        """Returns the speakers conditionings.

        TODO: Implement PASE embeddings as speaker representation. Check
            https://github.com/santi-pdp/pase

        Args:
            info: Metadata of the items, including utterance, speaker
                and dataset metadata.
            device: Identifier of the device where the returned Tensor
                should be moved.

        Returns:
            Tensor containing the speaker conditionings.
        """
        if self.kwargs['conds_speaker_type'] == 'embedding':
            speakers_ids = torch.tensor(
                [info_item['speaker']['index']
                 if info_item is not None
                 else 0 for info_item in info], dtype=torch.int64
            ).to(device)
            return self.speaker_embedding(speakers_ids).unsqueeze(1)
        else:
            raise NotImplementedError

    def _forward_content_conds(self, utt_conds):
        """Returns the content conditionings.

        Notice that for the case of acoustic conditionings, the Tensor
        returned from the dataset is already valid.

        Args:
            utt_conds: Tensor containing the content conditionings.

        Returns:
            Tensor containing the content conditionings.
        """
        if self.kwargs['conds_utterance_type'] == 'acoustic':
            return utt_conds

        embedded_features = []
        for i in [2, 3, 4, 5, 6]:
            embedded_features.append(
                self.conds_utt_phonemes_emb(utt_conds[:, :, i].long())
            )
        embedded_features.append(
            self.conds_utt_vowels_emb(utt_conds[:, :, 27].long())
        )
        for i in [31, 33, 41]:
            embedded_features.append(
                self.conds_utt_gpos_emb(utt_conds[:, :, i].long())
            )
        embedded_features.append(
            self.conds_utt_tobi_emb(utt_conds[:, :, 49].long())
        )
        embedded_features.append(utt_conds[:, :, 0:2])
        embedded_features.append(utt_conds[:, :, 7:27])
        embedded_features.append(utt_conds[:, :, 28:31])
        embedded_features.append(utt_conds[:, :, 32:33])
        embedded_features.append(utt_conds[:, :, 34:41])
        embedded_features.append(utt_conds[:, :, 42:49])
        embedded_features.append(utt_conds[:, :, 50:])
        return torch.cat(embedded_features, dim=2)


class FrameLevelLayer(torch.nn.Module):
    """Implementation of the Frame Level layer.

    Attributes:
        input_samples: Number of input samples of the layer.
        ratio: Upsampling ratio of the layer.
        rnn_layers: Number of layers used in ``self.rnn``.
        rnn_hidden_size: Hidden size of ``self.rnn``.
        x_expand: Instance of a ``torch.nn.Conv1d`` layer.
        conds_expand: Instance of a ``torch.nn.Conv1d`` layer.
        rnn: Instance of a ``torch.nn.GRU`` layer.
        rnn_h0: Instance of a ``torch.nn.Parameter``.
        upsample: Instance of a ``torch.nn.ConvTranspose1d`` layer.
        upsample_bias: Instance of a ``torch.nn.Parameter``.
    """

    def __init__(self, layer_n, **kwargs):
        super(FrameLevelLayer, self).__init__()
        self.input_samples = list(
            map(int, np.cumprod(kwargs['ratios']))
        )[layer_n]
        self.ratio = kwargs['ratios'][layer_n]
        self.rnn_layers = kwargs['rnn_layers'][layer_n]
        self.rnn_hidden_size = kwargs['rnn_hidden_size'][layer_n]

        self.x_expand = \
            torch.nn.Conv1d(self.input_samples, self.rnn_hidden_size, 1)
        self.conds_expand = \
            torch.nn.Conv1d(kwargs['conds_size'], self.rnn_hidden_size, 1)
        self.rnn = torch.nn.GRU(
            self.rnn_hidden_size, self.rnn_hidden_size, self.rnn_layers,
            batch_first=True
        )
        self.rnn_h0 = torch.nn.Parameter(
            torch.zeros(self.rnn_layers, self.rnn_hidden_size)
        )
        self.upsample = torch.nn.ConvTranspose1d(
            self.rnn_hidden_size, self.rnn_hidden_size, self.ratio,
            stride=self.ratio, bias=False
        )
        self.upsample_bias = \
            torch.nn.Parameter(torch.zeros(self.rnn_hidden_size, self.ratio))

        self._init_weights()

    def _init_weights(self):
        """Initializes the weights of the network.
        """
        torch.nn.init.kaiming_uniform_(self.x_expand.weight)
        torch.nn.init.constant_(self.x_expand.bias, 0)
        self.x_expand = torch.nn.utils.weight_norm(self.x_expand)

        torch.nn.init.kaiming_uniform_(self.conds_expand.weight)
        torch.nn.init.constant_(self.conds_expand.bias, 0)
        self.conds_expand = torch.nn.utils.weight_norm(self.conds_expand)

        self.upsample.reset_parameters()
        torch.nn.init.constant_(self.upsample_bias, 0)
        torch.nn.init.uniform_(
            self.upsample.weight,
            -np.sqrt(6 / self.rnn_hidden_size),
            np.sqrt(6 / self.rnn_hidden_size)
        )
        self.upsample = torch.nn.utils.weight_norm(self.upsample)

        for i in range(self.rnn_layers):
            torch.nn.init.constant_(
                getattr(self.rnn, 'bias_ih_l{}'.format(i)), 0
            )
            torch.nn.init.constant_(
                getattr(self.rnn, 'bias_hh_l{}'.format(i)), 0
            )
            FrameLevelLayer.concat_init(
                getattr(self.rnn, 'weight_ih_l{}'.format(i)),
                [FrameLevelLayer.lecun_uniform,
                 FrameLevelLayer.lecun_uniform, FrameLevelLayer.lecun_uniform]
            )
            FrameLevelLayer.concat_init(
                getattr(self.rnn, 'weight_hh_l{}'.format(i)),
                [FrameLevelLayer.lecun_uniform, FrameLevelLayer.lecun_uniform,
                 torch.nn.init.orthogonal_]
            )

    def forward(self, x, conds, upper_conditioning, rnn_state):
        """Forward pass through the Frame Level layer.

        Args:
            x: Tensor containing the samples used as input.
            conds: Tensor containing the hiddden representation that mixes
                speaker and content conditionings.
            upper_conditioning: Tensor containing the conditioning vector of
                the upper layer.
            rnn_state: Tensor containing the previous RNN state.

        Returns:
            Tuple of two positions containing:
                - Tensor containing the conditioning vector lower layers.
                - Tensor containing the new RNN state.
        """
        b, t, _ = x.size()
        if t != conds.shape[1]:
            upscale_ratio = int(x.shape[1] / conds.shape[1])
            conds = conds.unsqueeze(2) \
                .expand(b, conds.shape[1], upscale_ratio, conds.shape[2]) \
                .reshape(b, t, conds.shape[2])
        x = self.x_expand(x.permute(0, 2, 1)).permute(0, 2, 1)
        conds = self.conds_expand(conds.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + conds + upper_conditioning \
            if upper_conditioning is not None else x + conds
        hidden_state_tensor = torch.cat([
            self.rnn_h0.unsqueeze(1)
            if state is None else state.unsqueeze(1)
            for _, state in enumerate(rnn_state)
        ], dim=1)
        rnn_output, rnn_state_new = self.rnn(x, hidden_state_tensor)
        upsampling_bias = self.upsample_bias.unsqueeze(0).unsqueeze(2) \
            .expand(b, self.rnn_hidden_size, t, self.ratio) \
            .contiguous().view(b, self.rnn_hidden_size, t * self.ratio)
        upsampling_output = (
                self.upsample(rnn_output.permute(0, 2, 1)) + upsampling_bias
        ).permute(0, 2, 1)
        return upsampling_output, rnn_state_new

    @staticmethod
    def concat_init(tensor, inits):
        """Performs a custom initialization used in the package

        Args:
            tensor: Tensor to be initialized.
            inits: List of initialization functions to be applied.
        """
        tensor = tensor.data
        (length, fan_out) = tensor.size()
        fan_in = length // len(inits)
        chunk = tensor.new(fan_in, fan_out)
        for (i, init) in enumerate(inits):
            init(chunk)
            tensor[i * fan_in: (i + 1) * fan_in, :] = chunk

    @staticmethod
    def lecun_uniform(tensor):
        """Initializes a tensor using LeCun uniform initializer.

        Args:
            tensor: Tensor to be initialized.
        """
        fan_in = torch.nn.init._calculate_correct_fan(tensor, 'fan_in')
        torch.nn.init.uniform_(
            tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in)
        )


class SampleLevelLayer(torch.nn.Module):
    """Implementation of the Sample Level layer.

    Attributes:
        emb_layer: Instance of a ``torch.nn.Embedding`` layer.
        emb_layer_expand: Instance of a ``torch.nn.Conv1d`` layer.
        conds_expand: Instance of a ``torch.nn.Conv1d`` layer.
        comb_layer: Instance of a ``torch.nn.Linear`` layer.
        comb_layer_expand: Instance of a ``torch.nn.Conv1d`` layer.
        adapt: Instance of a ``torch.nn.Conv1d`` layer.
        input_samples: Number of input samples of the layer.
        kwargs: Dictionary containing the CLI arguments of the execution.
    """

    def __init__(self, **kwargs):
        super(SampleLevelLayer, self).__init__()
        self.emb_layer = torch.nn.Embedding(
            kwargs['q_levels'], kwargs['q_levels']
        )
        self.emb_layer_expand = torch.nn.Conv1d(
            kwargs['q_levels'], kwargs['rnn_hidden_size'][0],
            kwargs['ratios'][0], bias=False
        )
        self.conds_expand = torch.nn.Conv1d(
            kwargs['conds_size'], kwargs['rnn_hidden_size'][0], 1
        )
        self.comb_layer = torch.nn.Linear(
            kwargs['rnn_hidden_size'][0] * 3, kwargs['rnn_hidden_size'][0]
        )
        self.comb_layer_expand = torch.nn.Conv1d(
            kwargs['rnn_hidden_size'][0], kwargs['rnn_hidden_size'][0], 1
        )
        self.adapt = torch.nn.Conv1d(
            kwargs['rnn_hidden_size'][0], kwargs['q_levels'], 1
        )
        self._init_weights()

        self.input_samples = kwargs['ratios'][0]
        self.kwargs = kwargs

    def _init_weights(self):
        """Initializes the weights of the network.
        """
        torch.nn.init.kaiming_uniform_(self.emb_layer_expand.weight)
        self.emb_layer_expand = \
            torch.nn.utils.weight_norm(self.emb_layer_expand)

        torch.nn.init.kaiming_uniform_(self.comb_layer.weight)
        torch.nn.init.constant_(self.comb_layer.bias, 0)
        self.comb_layer_expand = \
            torch.nn.utils.weight_norm(self.comb_layer_expand)

        FrameLevelLayer.lecun_uniform(self.adapt.weight)
        torch.nn.init.constant_(self.adapt.bias, 0)
        self.adapt = torch.nn.utils.weight_norm(self.adapt)

    def forward(self, x, conds, upper_conditioning):
        """Forward pass through the Sample Level layer.

        Args:
            x: Tensor containing the samples used as input.
            conds: Tensor containing the hiddden representation that mixes
                speaker and content conditionings.
            upper_conditioning: Tensor containing the conditioning vector of
                the upper layer.

        Returns:
            Tensor containing the logits of the predicted samples.
        """
        upscale_ratio = int(upper_conditioning.shape[1] / conds.shape[1])
        conds = conds.unsqueeze(2).expand(
            x.size(0), conds.shape[1], upscale_ratio, conds.shape[2]
        ).reshape(x.size(0), upper_conditioning.shape[1], conds.shape[2])

        embedding_output = self.emb_layer(
            x.contiguous().view(-1)
        ).view(x.size(0), -1, self.kwargs['q_levels'])

        embedding_expand_output = self.emb_layer_expand(
            embedding_output.permute(0, 2, 1)
        )
        conds_expand_output = self.conds_expand(conds.permute(0, 2, 1))

        inputs_comb_output = F.relu(
            self.comb_layer(torch.cat((
                embedding_expand_output.permute(0, 2, 1),
                conds_expand_output.permute(0, 2, 1),
                upper_conditioning), dim=2)
            )
        )
        global_expand_output = F.relu(
            self.comb_layer_expand(inputs_comb_output.permute(0, 2, 1))
        )

        return self.adapt(global_expand_output).permute(0, 2, 1)
