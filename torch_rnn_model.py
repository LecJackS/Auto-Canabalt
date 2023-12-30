import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.sample_batch import SampleBatch

torch, nn = try_import_torch()

class TorchRNNModel(TorchRNN, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 #fc_size=32,
                 lstm_state_size=16,
                 fcnet_hiddens=[16,8],
                 drop_rate=0.5,
                 training=True):
        nn.Module.__init__(self)
        super(TorchRNNModel, self).__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        #self.fc_size = fc_size
        self.drop_rate = drop_rate
        self.training = training
        self.lstm_state_size = lstm_state_size

        # Build the Module from fc1 + fc2 + LSTM + 2xfc (action + value outs).
        self.fc_layers = nn.ModuleList([])
        for i in range(len(fcnet_hiddens)):
            fc_size = fcnet_hiddens[i]
            if i == 0:
                self.fc_layers.append(nn.Linear(self.obs_size, fc_size))
            else:
                self.fc_layers.append(nn.Linear(fcnet_hiddens[i-1], fc_size))
        self.lstm = nn.LSTM(
            fcnet_hiddens[-1], self.lstm_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.fc_layers[-1].weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.fc_layers[-1].weight.new(1, self.lstm_state_size).zero_().squeeze(0)
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        # print("Dimensions of:")
        # print("> self._features", self._features.shape)
        # print("> self.value_branch(self._features)", self.value_branch(self._features).shape)
        out = torch.reshape(self.value_branch(self._features), [-1])
        # print("> out.shape:", out.shape)

        return out

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        x = inputs
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = nn.functional.leaky_relu(x)
            x = nn.functional.dropout(x, p=self.drop_rate, training=self.training)

        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0),
                torch.unsqueeze(state[1], 0)])
        action_out = self.action_branch(self._features)

        # print("action_out.shape")
        # print(action_out.shape)
        # print(inputs.shape)
        # print(h.shape, c.shape)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]