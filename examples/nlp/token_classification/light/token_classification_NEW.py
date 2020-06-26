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

"""
Tutorial on how to use this script to solve NER task could be found here:
https://nvidia.github.io/NeMo/nlp/intro.html#named-entity-recognition
"""

import argparse
import os

import nemo.collections.nlp as nemo_nlp
import nemo.collections.nlp.utils.data_utils
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.collections.nlp.callbacks.token_classification_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.collections.nlp.data.datasets.datasets_utils.data_preprocessing import calc_class_weights
from nemo.collections.nlp.nm.data_layers import BertTokenClassificationDataLayer
from nemo.collections.nlp.nm.trainables import TokenClassifier
from nemo.utils.lr_policies import get_lr_policy
import pytorch_lightning as pl
from NER_model import TokenClassificationModel

# Parsing arguments
"""Provide extra arguments required for tasks."""
parser = argparse.ArgumentParser(description="Token classification with pretrained BERT")
parser.add_argument("--local_rank", default=None, type=int)

# training arguments
parser.add_argument(
    "--work_dir",
    default='output',
    type=str,
    help="The output directory where the model prediction and checkpoints will be written.",
)
parser.add_argument("--no_time_to_log_dir", action="store_true", help="whether to add time to work_dir or not")
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=5, type=int)
parser.add_argument("--amp_opt_level", default="O0", type=str, choices=["O0", "O1", "O2"])
parser.add_argument(
    "--save_epoch_freq",
    default=1,
    type=int,
    help="Frequency of saving checkpoint '-1' - step checkpoint won't be saved",
)
parser.add_argument(
    "--save_step_freq",
    default=-1,
    type=int,
    help="Frequency of saving checkpoint '-1' - step checkpoint won't be saved",
)
parser.add_argument(
    "--eval_step_freq", default=-1, type=int, help="Frequency of evaluation, -1 to evaluate every epoch"
)
parser.add_argument("--loss_step_freq", default=250, type=int, help="Frequency of printing loss")
parser.add_argument("--use_weighted_loss", action='store_true', help="Flag to indicate whether to use weighted loss")

# learning rate arguments
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--lr", default=5e-5, type=float)
parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--optimizer_kind", default="adam", type=str)

# task specific arguments
parser.add_argument("--fc_dropout", default=0.5, type=float)
parser.add_argument("--num_fc_layers", default=2, type=int)

# data arguments
parser.add_argument("--data_dir", default="/data", type=str)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--ignore_start_end", action='store_false')
parser.add_argument("--ignore_extra_tokens", action='store_false')
parser.add_argument("--none_label", default='O', type=str)
parser.add_argument("--mode", default='train_eval', choices=["train_eval", "train"], type=str)
parser.add_argument("--no_shuffle_data", action='store_false', dest="shuffle_data")
parser.add_argument("--use_cache", action='store_true', help="Whether to cache preprocessed data")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
parser.add_argument("--batches_per_step", default=1, type=int, help="Number of iterations per step.")
parser.add_argument(
    "--tokenizer",
    default="nemobert",
    type=str,
    choices=["nemobert", "sentencepiece"],
    help="tokenizer to use, only relevant when using custom pretrained checkpoint.",
)
parser.add_argument(
    "--vocab_file", default=None, type=str, help="Path to the vocab file. Required for pretrained Megatron models"
)
parser.add_argument(
    "--tokenizer_model",
    default=None,
    type=str,
    help="Path to pretrained tokenizer model, only used if --tokenizer is sentencepiece",
)
parser.add_argument(
    "--do_lower_case",
    action='store_true',
    help="Whether to lower case the input text. True for uncased models, False for cased models. "
    + "Only applicable when tokenizer is build with vocab file",
)

# model arguments
parser.add_argument(
    "--pretrained_model_name",
    default="bert-base-uncased",
    type=str,
    help="Name of the pre-trained model",
    choices=nemo_nlp.nm.trainables.get_pretrained_lm_models_list(),
)
parser.add_argument("--bert_checkpoint", default=None, type=str, help="Path to bert pretrained  checkpoint")
parser.add_argument("--bert_config", default=None, type=str, help="Path to bert config file in json format")


args = parser.parse_args()
logging.info(args)

if not os.path.exists(args.data_dir):
    raise FileNotFoundError(
        "Dataset not found. For NER, CoNLL-2003 dataset"
        "can be obtained at"
        "https://github.com/kyzhouhzau/BERT"
        "-NER/tree/master/data."
    )

model = TokenClassificationModel()

# Setup where your training data is
asr_model.setup_training_data(model_config['AudioToTextDataLayer'])
asr_model.setup_validation_data(model_config['AudioToTextDataLayer_eval'])

# trainer = pl.Trainer(val_check_interval=5, amp_level='O1', precision=16, gpus=2, max_epochs=50, distributed_backend='ddp')
trainer = pl.Trainer(val_check_interval=5)
trainer.fit(asr_model)

# Export for Jarvis
asr_model.save_to('qn.nemo', optimize_for_deployment=True)