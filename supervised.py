# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import argparse
from collections import OrderedDict
import torch

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator


VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'


# main
parser = argparse.ArgumentParser(description='Supervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=bool_flag, default=True, help="Export embeddings after training")
# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size")
# training refinement
parser.add_argument("--align_method", type=str, default="procrustes", help="Alignment method. Available 'procrustes', 'nn2'")
parser.add_argument("--n_iters", type=int, default=5, help="Number of iterations")
parser.add_argument("--epochs_per_iter", type=int, default=1, help="Number of epochs to train per iteration for non-procrustes methods")
parser.add_argument("--map_optimizer", type=str, default="adam,lr=0.001", help="Mapping optimizer for non-procrustes methods")
parser.add_argument("--map_batch_size", type=int, default=32, help="Mapping training batch size")
parser.add_argument("--map_most_frequent", type=int, default=0, help="Specify size of the dataset to use for training. 0 for the whole train dictionary.")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_train", type=str, default="default", help="Path to training dictionary (default: use identical character strings)")
parser.add_argument("--dico_train_sep", type=str, default=None, help="String separating source and target words in the training dictionary (assumed to be a space)")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=10000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default='', help="Reload source embeddings")
parser.add_argument("--src_emb_sep", type=str, default=" ", help="source embeddings separator")
parser.add_argument("--src_emb_vocab", type=str, default="", help="source embeddings vocab file, if src_emb is a .bin file")
parser.add_argument("--tgt_emb", type=str, default='', help="Reload target embeddings")
parser.add_argument("--tgt_emb_sep", type=str, default=" ", help="Target embeddings separator")
parser.add_argument("--tgt_emb_vocab", type=str, default="", help="target embeddings vocab file, if tgt_emb is a .bin file")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")


# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert params.dico_train in ["identical_char", "default"] or os.path.isfile(params.dico_train)
assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping, _ = build_model(params, False)
trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
evaluator = Evaluator(trainer)

# load a training dictionary. if a dictionary path is not provided, use a default
# one ("default") or create one based on identical character strings ("identical_char")
trainer.load_training_dico(params.dico_train, sep=params.dico_train_sep)


def evaluate_and_save():
    # embeddings evaluation
    to_log = OrderedDict({'n_iter': n_iter})
    evaluator.all_eval(to_log)

    # JSON log / save best model / end of epoch
    logger.info("__log__:%s" % json.dumps(to_log))
    trainer.save_best(to_log, VALIDATION_METRIC)
    logger.info('End of refinement iteration %i.\n\n' % n_iter)


if params.align_method == 'procrustes':
    """
    Learning loop for Procrustes Iterative Refinement
    """
    for n_iter in range(params.n_iters):
        logger.info('Starting refinement iteration %i...' % n_iter)

        # build a dictionary from aligned embeddings (unless
        # it is the first iteration and we use the init one)
        if n_iter > 0 or not hasattr(trainer, 'dico'):
            trainer.build_dictionary()

        # apply the Procrustes solution
        trainer.procrustes()
        evaluate_and_save()
else:
    """
    Learning loop for standard ML training
    """
    for n_iter in range(params.n_iters):
        for n_epoch in range(params.epochs_per_iter):
            logger.info('Iter %i epoch %i... ' % (n_iter, n_epoch))

            trainer.train_mapping_epoch_from_dico()
        evaluate_and_save()

# export embeddings to a text format
if params.export:
    trainer.reload_best()
    trainer.export()
