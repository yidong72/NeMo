# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from torch import nn as nn

from nemo.backends.pytorch import MultiLayerPerceptron, TrainableNM
from nemo.collections.nlp.utils.transformer_utils import transformer_weights_init
from nemo.core import ChannelType, LogitsType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['PunctCapitTokenClassifier']


class PunctCapitTokenClassifier(TrainableNM):
    """
    Neural module which consists of MLP followed by softmax classifier for each
    token in the sequence.

    Args:
        hidden_size (int): hidden size (d_model) of the Transformer
        num_classes (int): number of classes in softmax classifier, e.g. size
            of the vocabulary in language modeling objective
        num_layers (int): number of layers in classifier MLP
        activation (str): activation function applied in classifier MLP layers
        log_softmax (bool): whether to apply log_softmax to MLP output
        dropout (float): dropout ratio applied to MLP
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {"hidden_states": NeuralType(('B', 'T', 'C'), ChannelType())}

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            "punct_logits": NeuralType(('B', 'T', 'D'), LogitsType()),
            "capit_logits": NeuralType(('B', 'T', 'D'), LogitsType()),
            "part_sent_logits": NeuralType(('B', 'D'), LogitsType(), optional=True),
        }

    def __init__(
        self,
        hidden_size,
        punct_num_classes,
        capit_num_classes,
        punct_num_layers=2,
        capit_num_layers=2,
        activation='relu',
        log_softmax=True,
        dropout=0.0,
        use_transformer_pretrained=True,
        part_sent_num_layers=2,
        add_part_sent_head=False
    ):
        # Pass name up the module class hierarchy.
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.punct_mlp = MultiLayerPerceptron(
            hidden_size, punct_num_classes, self._device, punct_num_layers, activation, log_softmax
        )
        self.capit_mlp = MultiLayerPerceptron(
            hidden_size, capit_num_classes, self._device, capit_num_layers, activation, log_softmax
        )
        if add_part_sent_head:
            self.part_sent_mlp = MultiLayerPerceptron(
            hidden_size, 2, self._device, part_sent_num_layers, activation, log_softmax
        )
        self.add_part_sent_head = add_part_sent_head
        if use_transformer_pretrained:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))
        # self.to(self._device) # sometimes this is necessary

    def __str__(self):
        return self.name

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        punct_logits = self.punct_mlp(hidden_states)
        capit_logits = self.capit_mlp(hidden_states)
        output = [punct_logits, capit_logits]
        if self.add_part_sent_head:
            output.append(self.part_sent_mlp(hidden_states[:, 0]))
        return output
