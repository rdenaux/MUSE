# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn

from .utils import load_external_embeddings, normalize_embeddings


class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.emb_dim = params.emb_dim
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)


class NNMapper(nn.Module):
    def __init__(self, input_dim, hidden_dims,
                 output_dim=None, hidden_node_dropout_ps=None):
        super(NNMapper, self).__init__()
        self.hidden_layer_cnt = len(hidden_dims)
        output_dim = output_dim if output_dim else input_dim
        fcs = []
        dropouts = []
        if hidden_node_dropout_ps:
            assert(len(hidden_node_dropout_ps) == self.hidden_layer_cnt)
        else:
            hidden_node_dropout_ps = [.5 for x in range(self.hidden_layer_cnt)]
        for hli in range(self.hidden_layer_cnt + 1):
            in_dim = input_dim if hli == 0 else hidden_dims[hli - 1]
            out_dim = output_dim if hli == self.hidden_layer_cnt else hidden_dims[hli]
            fcs.append(nn.Linear(in_dim, out_dim))
            if (hli < self.hidden_layer_cnt):
                dropouts.append(nn.Dropout(p=hidden_node_dropout_ps[hli]))
        print('fcs %d, dropouts %d' % (len(fcs), len(dropouts)))
        self.fcs = nn.ModuleList(fcs)
        self.dropouts = nn.ModuleList(dropouts)

    def forward(self, x):
        for hli in range(self.hidden_layer_cnt):
            x = self.dropouts[hli](nn.functional.relu(self.fcs[hli](x)))
        return self.fcs[self.hidden_layer_cnt](x)


def build_model(params, with_dis):
    """
    Build all components of the model.
    """
    # source embeddings
    src_dico, _src_emb = load_external_embeddings(params, source=True)
    params.src_dico = src_dico
    src_emb = nn.Embedding(len(src_dico), params.emb_dim, sparse=True)
    src_emb.weight.data.copy_(_src_emb)

    # target embeddings
    if params.tgt_lang:
        tgt_dico, _tgt_emb = load_external_embeddings(params, source=False)
        params.tgt_dico = tgt_dico
        tgt_emb = nn.Embedding(len(tgt_dico), params.emb_dim, sparse=True)
        tgt_emb.weight.data.copy_(_tgt_emb)
    else:
        tgt_emb = None

    # mapping
    if params.align_method == 'procrustes':
        mapping = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
        if getattr(params, 'map_id_init', True):
            mapping.weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))
    elif params.align_method == 'nn2':
        hidden_dims = [6500, 6500]
        mapping = NNMapper(params.emb_dim, hidden_dims)

    # discriminator
    discriminator = Discriminator(params) if with_dis else None

    # cuda
    if params.cuda:
        src_emb.cuda()
        if params.tgt_lang:
            tgt_emb.cuda()
        mapping.cuda()
        if with_dis:
            discriminator.cuda()

    # normalize embeddings
    normalize_embeddings(src_emb.weight.data, params.normalize_embeddings)
    if params.tgt_lang:
        normalize_embeddings(tgt_emb.weight.data, params.normalize_embeddings)

    return src_emb, tgt_emb, mapping, discriminator
