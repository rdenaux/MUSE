# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
import numpy as np
import torch
import re

from ..utils import get_nn_avg_dist


DIC_EVAL_PATH = 'data/crosslingual/dictionaries/'


logger = getLogger()


def load_identical_char_dico(word2id1, word2id2):
    """
    Build a dictionary of identical character strings.
    """
    pairs = [(w1, w1) for w1 in word2id1.keys() if w1 in word2id2]
    if len(pairs) == 0:
        raise Exception("No identical character strings were found. "
                        "Please specify a dictionary.")

    logger.info("Found %i pairs of identical character strings." % len(pairs))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico


def _load_dictionary_pairs(path, word2id11, word2id2, sep=None):
    """
    Return a list of word pairs as they appear in the input file at path.
    Pairs in the file which do not appear in the src or tgt vocabularies are 
    removed, but logged.
    """
    assert os.path.isfile(path), 'path is not a file'

    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with open(path, 'r') as f:
        for _, line in enumerate(f):
            # assert line == line.lower()
            word1, word2 = line.rstrip().split(sep=sep)
            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    logger.info("Found %i pairs of words in the dictionary %s (%i unique). "
                "%i other pairs contained at least one unknown word "
                "(%i in lang1, %i in lang2)"
                % (len(pairs), path, len(set([x for x, _ in pairs])),
                   not_found, not_found1, not_found2))
    return pairs

def load_dictionary(path, word2id1, word2id2, sep=None):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """
    pairs = _load_dictionary_pairs(path, word2id1, word2id2, sep=sep)
    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico


def get_word_translation_accuracy(lang1, word2id1, emb1, lang2, word2id2, emb2, method):
    """
    Given source and target word embeddings, and a dictionary,
    evaluate the translation accuracy using the precision@k.
    """
    path = os.path.join(DIC_EVAL_PATH, '%s-%s.5000-6500.txt' % (lang1, lang2))
    return _get_translation_acc(lang1, word2id1, emb1, lang2, word2id2, emb2, method, path)


def get_syncon_translation_accuracy(lang1, word2id1, emb1, lang2, word2id2, emb2, method):
    syn_tests = [f for f in os.listdir(os.path.join(DIC_EVAL_PATH))
                 if re.match(r'%s-%s\.syn-test.*\.txt' % (lang1, lang2), f)]
    result = []
    for f in syn_tests:
        path = os.path.join(DIC_EVAL_PATH, f)
        label = f[:-4]
        result.extend(_get_translation_acc(
            lang1, word2id1, emb1, lang2, word2id2, emb2,
            method, path, label=label, dict_sep=':'))
    return result

def get_lemma_translation_accuracy(lang1, word2id1, emb1, lang2, word2id2, emb2, method):
    lem_tests = [f for f in os.listdir(os.path.join(DIC_EVAL_PATH))
                 if re.match(r'%s-%s\.lem-test.*\.txt' % (lang1, lang2), f)]
    result = []
    for f in lem_tests:
        path = os.path.join(DIC_EVAL_PATH, f)
        label = f[:-4]
        result.extend(_get_translation_acc(
            lang1, word2id1, emb1, lang2, word2id2, emb2,
            method, path, label=label, dict_sep=':'))
    return result


def _get_translation_acc(lang1, word2id1, emb1, lang2, word2id2, emb2, method, path, label=None, dict_sep=None):
    try:
        dico = load_dictionary(path, word2id1, word2id2, sep=dict_sep)
    except AssertionError as e:
        msg = 'Error loading translation dictionary %s: %s ' % (path, e)
        logger.error(msg)
        return [msg]

    dico = dico.cuda() if emb1.is_cuda else dico

    assert dico[:, 0].max() < emb1.size(0)
    assert dico[:, 1].max() < emb2.size(0)

    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

    # nearest neighbors
    if method == 'nn':
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))

    # inverted softmax
    elif method.startswith('invsm_beta_'):
        beta = float(method[len('invsm_beta_'):])
        bs = 128
        word_scores = []
        for i in range(0, emb2.size(0), bs):
            scores = emb1.mm(emb2[i:i + bs].transpose(0, 1))
            scores.mul_(beta).exp_()
            scores.div_(scores.sum(0, keepdim=True).expand_as(scores))
            word_scores.append(scores.index_select(0, dico[:, 0]))
        scores = torch.cat(word_scores, 1)

    # contextual dissimilarity measure
    elif method.startswith('csls_knn_'):
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        # queries / scores
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[dico[:, 0]][:, None] + average_dist2[None, :])

    else:
        raise Exception('Unknown method: "%s"' % method)

    results = []
    top_matches = scores.topk(100, 1, True)[1]
    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1)
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0]):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        logger.info("%i source words - %s - Precision at k = %i: %f" %
                    (len(matching), method, k, precision_at_k))
        prefix = '%s_' % label if label else ''
        results.append(('%sprecision_at_%i' % (prefix, k), precision_at_k))

    return results
