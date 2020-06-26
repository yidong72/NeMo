# -*- coding: utf-8 -*-

from typing import Dict, Optional

# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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
import torch
from pytorch_lightning import LightningModule
import nemo
from nemo.collections.nlp.nm.trainables.common.huggingface.bert_nm import BERT
from nemo.core.apis import NeuralModelAPI, NeuralModuleAPI
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.collections.nlp.nm.trainables import TokenClassifier

class TokenClassificationModel(LightningModule, NeuralModelAPI):
    @classmethod
    def from_cloud(cls, name: str):
        pass

    def __init__(
        self,
        pretrained_model_name='bert-base-uncased',
        tokenizer='nemobert',
        tokenizer_model=None,
        vocab_file=None,
        do_lower_case=False,
        dropout=0.1,
        num_layers=2,
        num_classes=9,
    ):
        super().__init__()
        bert_model = BERT(pretrained_model_name='bert-base-uncased')
        self.encoder = bert_model.bert
        tokenizer = nemo.collections.nlp.data.tokenizers.get_tokenizer(
            tokenizer_name=tokenizer,
            pretrained_model_name=pretrained_model_name,
            tokenizer_model=tokenizer_model,
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
        )
      
        self.classifier = TokenClassifier(hidden_size=self.encoder.hidden_size, num_classes=num_layers, dropout=dropout, num_layers=num_layers)
        self.loss = CrossEntropyLossNM(logits_ndim=3)
        

        self.__train_dl = None
        self.__val_dl = None
        self.__optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)

    def forward(self, input_ids, input_type_ids, input_mask, subtokens_mask):
        hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        logits = self.classifier(hidden_states=hidden_states)
        return logits

    def save_to(self, save_path: str, optimize_for_deployment=False):
        print("TODO: Implement Me")

    def restore_from(cls, restore_path: str):
        print("TODO: Implement Me")

    def training_step(self, batch, batch_nb):
        self.train()
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, labels = batch
        logits = self.forward(input_ids, input_type_ids, input_mask, subtokens_mask)
        train_loss = self.loss._loss_function(logits=logits, labels=labels, loss_mask=loss_mask)
        return {'loss': train_loss}

    def configure_optimizers(self):
        return self.__optimizer

    def train_dataloader(self):
        return self.__train_dl.data_loader

    def setup_training_data(self, train_data_layer_params):
        """
        Setups data loader to be used in training
        Args:
            train_data_layer_params: training data layer parameters.
        Returns:

        """
        text_file,
        label_file,
        tokenizer,
        max_seq_length,
        pad_label='O',
        label_ids=None,
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        ignore_extra_tokens=False,
        ignore_start_end=False,
        use_cache=False,
        dataset_type=BertTokenClassificationDataset,
    ):
        dataset_params = {
            'text_file': text_file,
            'label_file': label_file,
            'max_seq_length': max_seq_length,
            'tokenizer': tokenizer,
            'num_samples': num_samples,
            'pad_label': pad_label,
            'label_ids': label_ids,
            'ignore_extra_tokens': ignore_extra_tokens,
            'ignore_start_end': ignore_start_end,
            'use_cache': use_cache,
        }
        super().__init__(dataset_type, dataset_params, batch_size, shuffle)
        self.__train_dl = NeuralModuleAPI.from_config(train_data_layer_params)

    def validation_step(self, batch, batch_idx):
        self.eval()
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, labels = batch
        logits = self.forward(input_ids, input_type_ids, input_mask, subtokens_mask)
        eval_loss = self.loss._loss_function(logits=logits, labels=labels, loss_mask=loss_mask)
        return {'loss': eval_loss, 'logits':logits, 'labels':labels, 'subtokens_mask':subtokens_mask}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        import pdb; pdb.set_trace()
        # labels = np.asarray(global_vars['all_labels'])
        # preds = np.asarray(global_vars['all_preds'])
        # subtokens_mask = np.asarray(global_vars['all_subtokens_mask']) > 0.5

        # labels = labels[subtokens_mask]
        # preds = preds[subtokens_mask]

        # # print predictions and labels for a small random subset of data
        # sample_size = 20
        # i = 0
        # if preds.shape[0] > sample_size + 1:
        #     i = random.randint(0, preds.shape[0] - sample_size - 1)
        # logging.info("Sampled preds: [%s]" % list2str(preds[i : i + sample_size]))
        # logging.info("Sampled labels: [%s]" % list2str(labels[i : i + sample_size]))

        # accuracy = sum(labels == preds) / labels.shape[0]
        # logging.info(f'Accuracy: {accuracy}')

        return {'val_loss': val_loss_mean}

    def val_dataloader(self):
        return self.__val_dl.data_loader

    def setup_validation_data(self, val_data_layer_params):
        """
        Setups data loader to be used in validation
        Args:
            val_data_layer_params: validation data layer parameters.
        Returns:

        """
        self.__val_dl = NeuralModuleAPI.from_config(val_data_layer_params)

    def setup_optimizer(self, optimizer_params):
        self.__optimizer = torch.optim.Adam(self.parameters(), lr=optimizer_params['lr'])