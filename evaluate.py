# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# python evaluate.py --crosslingual --src_lang en --tgt_lang es --src_emb data/wiki.en-es.en.vec --tgt_emb data/wiki.en-es.es.vec

import os
import argparse
from collections import OrderedDict

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator

# main
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
# data
parser.add_argument("--src_lang", type=str, default="", help="Source language")
parser.add_argument("--tgt_lang", type=str, default="", help="Target language")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--src_emb_sep", type=str, default=" ", help="source embeddings separator")
parser.add_argument("--src_emb_vocab", type=str, default="", help="source embeddings vocab file, if src_emb is a .bin file")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--tgt_emb_sep", type=str, default=" ", help="Target embeddings separator")
parser.add_argument("--tgt_emb_vocab", type=str, default="", help="target embeddings vocab file, if tgt_emb is a .bin file")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
# reload pre-trained mapping
parser.add_argument("--mapping", type=str, default="", help="Reload pre-trained mapping")


# parse parameters
params = parser.parse_args()

# check parameters
assert params.src_lang, "source language undefined"
assert os.path.isfile(params.src_emb)
assert not params.tgt_lang or os.path.isfile(params.tgt_emb)

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping, _ = build_model(params, False)
trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
trainer.reload_best_from(params.mapping)
evaluator = Evaluator(trainer)

# run evaluations
to_log = OrderedDict({'n_iter': 0})
evaluator.monolingual_wordsim(to_log)
if params.tgt_lang:
    evaluator.crosslingual_wordsim(to_log)
    evaluator.word_translation(to_log)
    evaluator.sent_translation(to_log)
    evaluator.syncon_translation(to_log)
    # evaluator.dist_mean_cosine(to_log)
