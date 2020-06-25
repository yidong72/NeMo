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

import argparse
import os
from nemo.collections.nlp.utils.data_utils import get_vocab
from nemo.utils import logging

parser = argparse.ArgumentParser(description='Punctuation and capitalization detection error analysis')
parser.add_argument("--input_file", default='text_dev.txt', type=str,
    help='Path to file formatted as text_train.txt, see documentation')
parser.add_argument("--labels_preds_dir", default='error_analysis/', type=str, 
    help='Path to dir that contains task_name_labels_preds.txt files')
parser.add_argument(
    "--labels_dict_dir",
    default='data_dir',
    type=str,
    help='Path to directory with punct_label_ids.csv, capit_label_ids.csv and part_sent_label_ids.csv(optional) files. ' +
    'These files are generated during training when the datalayer is created',
)
parser.add_argument(
    "--labels_dict_dir",
    default='data_dir',
    type=str,
    help='Path to directory with punct_label_ids.csv, capit_label_ids.csv and part_sent_label_ids.csv(optional) files. ' +
    'These files are generated during training when the datalayer is created',
)
parser.add_argument("--none_label", default='O', type=str, help='None label')

args = parser.parse_args()

punct_labels_dict_path = os.path.join(args.labels_dict_dir, 'punct_label_ids.csv')
capit_labels_dict_path = os.path.join(args.labels_dict_dir, 'capit_label_ids.csv')

if not os.path.exists(punct_labels_dict_path) or not os.path.exists(capit_labels_dict_path):
    raise ValueError ('--labels_dict_dir should contain punct_label_ids.csv and capit_label_ids.csv generated during training')

punct_labels_dict = get_vocab(punct_labels_dict_path)
capit_labels_dict = get_vocab(capit_labels_dict_path)

task_names = ['punct', 'capit']
labels_preds_dict = {}
for task_name in task_names:
    labels_preds_dict[task_name] = {}
    with open(os.path.join(args.labels_preds_dir, task_name + '_labels_preds.txt'), 'r') as f:
        labels_preds_dict[task_name]['labels'], labels_preds_dict[task_name]['preds'] = f.readlines()

def _get_combined_output(words, mode='preds'):
    output = ''
    for j, w in enumerate(words):
        punct_label = punct_labels_dict[labels_preds_dict['punct'][mode][j]]
        capit_label = capit_labels_dict[labels_preds_dict['capit'][mode][j]]
        if capit_label != args.none_label:
            w = w.capitalize()
        output += w
        if punct_label != args.none_label:
            output += punct_label
        output += ' '
    return output

correct = 0
wrong = 0
with open(args.input_file, 'w') as text:
    for i, line in enumerate(text):
        words = line.stirp().split()

        prediction = _get_combined_output(words, mode='preds').strip()
        ground_truth = _get_combined_output(words, mode='labels').strip()
        if prediction == ground_truth:
            correct += 1
        else:
            wrong += 1
            logging.info(f'Predictions: {prediction.strip()}\n')
            logging.info(f'Ground Truth: {prediction.strip()}\n')

logging.into(f'Number of correct predictions: {correct}')
logging.into(f'Number of wrong predictions: {wrong}')
