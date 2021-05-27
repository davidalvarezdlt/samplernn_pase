import samplernn_pase.model
import samplernn_pase.optimizer
import skeltorch
import torch
import torch.optim.lr_scheduler


class SampleRNNPASERunner(skeltorch.Runner):
    scheduler = None

    def init_model(self, device):
        self.model = samplernn_pase.model.SampleRNNModel(
            conds_speaker_type=self.experiment.configuration.get('conditionals', 'conds_speaker_type'),
            conds_speaker_n=len(self.experiment.data.speakers_info),
            conds_speaker_size=self.experiment.configuration.get('conditionals', 'conds_speaker_size'),
            conds_utterance_type=self.experiment.configuration.get('conditionals', 'conds_utterance_type'),
            conds_utterance_linguistic_n=self.experiment.data.get_conds_linguistic_size(),
            conds_utterance_linguistic_emb_size=
            self.experiment.configuration.get('conditionals', 'conds_utterance_linguistic_emb_size'),
            conds_size=self.experiment.configuration.get('conditionals', 'conds_size'),
            sequence_length=self.experiment.configuration.get('model', 'sequence_length'),
            ratios=self.experiment.configuration.get('model', 'ratios'),
            rnn_layers=self.experiment.configuration.get('model', 'rnn_layers'),
            rnn_hidden_size=self.experiment.configuration.get('model', 'rnn_hidden_size'),
            q_type_ulaw=self.experiment.configuration.get('model', 'q_type_ulaw'),
            q_levels=self.experiment.configuration.get('model', 'q_levels')
        ).to(device)

    def init_optimizer(self, device):
        self.optimizer = samplernn_pase.optimizer.AdamClipped(
            self.model.parameters(), lr=self.experiment.configuration.get('training', 'lr')
        )

    def init_others(self, device):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min',
            patience=self.experiment.configuration.get('training', 'lr_scheduler_patience'),
            factor=self.experiment.configuration.get('training', 'lr_scheduler_factor')
        )

    def load_states_others(self, checkpoint_data):
        self.scheduler.load_state_dict(checkpoint_data['scheduler'])

    def save_states_others(self):
        return {'scheduler': self.scheduler.state_dict()}

    def train_step(self, it_data, device):
        self.test(None, device)
        x, y, utt_conds, reset, info = it_data
        x, y, utt_conds = x.to(device), y.to(device), utt_conds.to(device)
        y_hat, y = self.model(x, y, utt_conds, info, reset)
        return torch.nn.functional.nll_loss(y_hat.view(-1, y_hat.size(2)), y.view(-1))

    def train_before_epoch_tasks(self, device):
        super().train_before_epoch_tasks(device)
        self.experiment.tbx.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.counters['epoch'])

    def train_after_epoch_tasks(self, device):
        self.scheduler.step(self.losses_epoch['validation'][self.counters['epoch']], self.counters['epoch'])
        self.test(None, device)

    def test(self, epoch, device):
        # Check if test has a forced epoch to load objects and restore checkpoint
        if epoch is not None and epoch not in self.experiment.checkpoints_get():
            exit('Epoch {} not found.'.format(epoch))
        elif epoch is not None:
            self.load_states(epoch, device)

        # Iterate over the test samples and save them in TensorBoard
        for it_data in self.experiment.data.loaders['test']:
            x, utt_conds, info = it_data
            x, utt_conds = x.to(device), utt_conds.to(device)
            with torch.no_grad():
                y_hat = self.model.test(utt_conds, info)
            self.experiment.tbx.add_audio(info['utterance']['name'][0], y_hat, self.counters['epoch'], 16000)
