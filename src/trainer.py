# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
import numpy as np
import scipy
import scipy.linalg
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.autograd import Variable
from torch.nn import functional as F

from .utils import get_optimizer, export_embeddings
from .utils import clip_parameters
from .dico_builder import build_dictionary
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary


logger = getLogger()


class Trainer(object):

    def __init__(self, src_emb, tgt_emb, mapping, discriminator, params):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        self.mapping = mapping
        self.discriminator = discriminator
        self.params = params

        # optimizers
        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            self.map_optimizer = optim_fn(mapping.parameters(), **optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_params)
        else:
            assert discriminator is None

        # best validation score
        self.best_valid_metric = -1e12

        self.decrease_lr = False
        self.map_trainloader = None

    def _get_map_xy(self):
        """
        Get the training dictionary as input / output targets
        """
        mf = self.params.map_most_frequent
        assert mf <= self.dico.shape[0]
        ds_size = self.dico.shape[0] if mf == 0 else mf
        src_ids = self.dico[:ds_size, 0]
        tgt_ids = self.dico[:ds_size, 1]
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        src_emb = self.src_emb(Variable(src_ids, volatile=True))
        tgt_emb = self.tgt_emb(Variable(tgt_ids, volatile=True))
        return src_emb, tgt_emb

    def get_map_train_loader(self):
        if self.map_trainloader:
            return self.map_trainloader
        x, y = self._get_map_xy()
        tds = TensorDataset(x, y)
        self.map_trainloader = DataLoader(
            tds,
            batch_size=self.params.map_batch_size,
            sampler=RandomSampler(tds),
            num_workers=2)
        return self.map_trainloader

    def get_dis_xy(self, volatile):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        assert mf <= min(len(self.src_dico), len(self.tgt_dico))
        src_ids = torch.LongTensor(bs).random_(len(self.src_dico) if mf == 0 else mf)
        tgt_ids = torch.LongTensor(bs).random_(len(self.tgt_dico) if mf == 0 else mf)
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        src_emb = self.src_emb(Variable(src_ids, volatile=True))
        tgt_emb = self.tgt_emb(Variable(tgt_ids, volatile=True))
        src_emb = self.mapping(Variable(src_emb.data, volatile=volatile))
        tgt_emb = Variable(tgt_emb.data, volatile=volatile)

        # input / target
        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.FloatTensor(2 * bs).zero_()
        y[:bs] = 1 - self.params.dis_smooth
        y[bs:] = self.params.dis_smooth
        y = Variable(y.cuda() if self.params.cuda else y)

        return x, y

    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()

        # loss
        x, y = self.get_dis_xy(volatile=True)
        preds = self.discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y)
        stats['DIS_COSTS'].append(loss.data[0])

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (discriminator)")
            exit()

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        clip_parameters(self.discriminator, self.params.dis_clip_weights)

    def mapping_step(self, stats):
        """
        Fooling discriminator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        self.discriminator.eval()

        # loss
        x, y = self.get_dis_xy(volatile=False)
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
        loss = self.params.dis_lambda * loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (fool discriminator)")
            exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()
        self.orthogonalize()

        return 2 * self.params.batch_size

    def train_mapping_epoch(self):
        """
        Train mapping (non-adversarially)
        """
        criterion = torch.nn.CosineEmbeddingLoss()
        loader = self.get_map_train_loader()
        self.map_optimizer.zero_grad()
        self.mapping.train(True)
        for i, data in enumerate(loader, 0):
            src_embs, tgt_embs = data
            if self.params.cuda:
                src_embs, tgt_embs = src_embs.cuda(), tgt_embs.cuda()
            outputs = self.mapping(src_embs)
            loss = criterion(outputs, tgt_embs)
            loss.backward()
            self.map_optimizer.step()
            if i % 100 == 0:
                logger.info("Step %i with loss %d" % (i, loss))

    def train_mapping_epoch_from_dico(self):
        criterion = torch.nn.CosineEmbeddingLoss(size_average=False)
        self.map_optimizer.zero_grad()
        mf = self.params.map_most_frequent
        assert mf <= self.dico.shape[0]
        ds_size = self.dico.shape[0] if mf == 0 else mf
        perm = np.random.permutation(ds_size)
        bs = self.params.map_batch_size
        cnt = 0
        self.mapping.train(True)
        for s in range(0, ds_size, bs):
            e = min(len(perm), s+bs)
            bis = torch.LongTensor(perm[s:e])
            if self.params.cuda:
                bis = bis.cuda()
            dico_batch = self.dico.index_select(0, bis)
            # logger.info('dico_batch size %s ' % str(dico_batch.size()))
            src_ids = dico_batch[:, 0]
            tgt_ids = dico_batch[:, 1]
            # logger.info('src_ids %s, min %i max %i'% (src_ids.size(), src_ids.min(), src_ids.max()))
            # all training pairs are positive examples
            tgts = torch.LongTensor(e-s).fill_(1)
            # logger.info('tgts size %s' % (tgts.size()))
            if self.params.cuda:
                src_ids = src_ids.cuda()
                tgt_ids = tgt_ids.cuda()
                tgts = tgts.cuda()

            src_embs = self.src_emb(Variable(src_ids, requires_grad=False))
            tgt_embs = self.tgt_emb(Variable(tgt_ids, requires_grad=False))
            if self.params.cuda:
                src_embs, tgt_embs = src_embs.cuda(), tgt_embs.cuda()
            outputs = self.mapping(src_embs)
            #logger.info('outputs %s, expected %s, tgts %s' % (outputs.size(), tgt_embs.size(), tgts.size()))
            #logger.info('outputs[0,0:3] %s, expected[0,0:3] %s' % (outputs[:2,0:3].data, tgt_embs[:2,0:3].data))
            loss = criterion(outputs, tgt_embs, Variable(tgts))
            # logger.info("Step %i with loss %f" % (cnt, loss))
            loss.backward()
            self.map_optimizer.step()
            if cnt % 100 == 0:
                logger.info("Step %i with loss %f" % (cnt, loss))
            cnt = cnt + 1

    def load_training_dico(self, dico_train, sep=None):
        """
        Load training dictionary.
        """
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

        # identical character strings
        if dico_train == "identical_char":
            self.dico = load_identical_char_dico(word2id1, word2id2)
        # use one of the provided dictionary
        elif dico_train == "default":
            filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
            self.dico = load_dictionary(
                os.path.join(DIC_EVAL_PATH, filename),
                word2id1, word2id2
            )
        # dictionary provided by the user
        else:
            self.dico = load_dictionary(dico_train, word2id1, word2id2, sep=sep)

        # cuda
        if self.params.cuda:
            self.dico = self.dico.cuda()

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico = build_dictionary(src_emb, tgt_emb, self.params)

    def procrustes(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]
        W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        if self.params.map_beta > 0:
            W = self.mapping.weight.data
            beta = self.params.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.map_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True

    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            # save the mapping
            path = os.path.join(self.params.exp_path, 'best_mapping.t7')
            if self.params.align_method == 'procrustes':
                W = self.mapping.weight.data.cpu().numpy()
                logger.info('* Saving the mapping to %s ...' % path)
                torch.save(W, path)
            elif self.params.align_method == 'nn2':
                torch.save(self.mapping.state_dict(), path)
            else:
                raise RuntimeError("invalid align_method %s" % self.params.align_method)

    def reload_best(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params.exp_path, 'best_mapping.t7')
        self.reload_best_from(path)

    def reload_best_from(self, path):
        logger.info('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        if self.params.align_method == 'procrustes':
            to_reload = torch.from_numpy(torch.load(path))
            W = self.mapping.weight.data
            assert to_reload.size() == W.size()
            W.copy_(to_reload.type_as(W))
        elif self.params.align_method == 'nn2':
            self.mapping.load_state_dict(torch.load(path))
        else:
            raise RuntimeError("invalid align_method %s" % self.params.align_method)

    def export(self):
        """
        Export embeddings to a text file.
        """
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        export_embeddings(src_emb.cpu().numpy(), tgt_emb.cpu().numpy(), self.params)
