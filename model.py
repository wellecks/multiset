from collections import defaultdict
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
import util
from rnn import ConvLSTM


class Model(nn.Module):
    def __init__(self, cnn, num_classes, use_cuda, sample):
        super(Model, self).__init__()
        self._cnn = cnn
        hidden_channels = self._cnn.asize[0]
        self._rnn = ConvLSTM(self._cnn.asize[0]+1, hidden_channels)

        embed_size = self._cnn.asize[1]*self._cnn.asize[2]
        self._clf = nn.Linear(self._cnn.asize[0]+embed_size, num_classes)
        self._class_embed = nn.Embedding(num_classes, embed_size)
        self._stop_conv = nn.Conv2d(self._cnn.asize[0], 1, kernel_size=3, padding=1)
        self._stop = nn.Linear(self._cnn.asize[1]*self._cnn.asize[2], 1)
        self._sample = sample
        self._num_classes = num_classes

    def forward(self, x, labels):
        path = defaultdict(list)
        A = self._cnn.activations(x)
        h = Variable(util.cuda_as(th.zeros(A.size()), A))
        c = Variable(util.cuda_as(th.zeros(A.size()), A))
        prev_class = Variable(util.cuda_as(th.zeros(x.size(0)), x).long())

        if self._sample == 'oracle':
            samples = util.sample_permutation(labels, self._num_classes)

        for t in range(labels.size(1)):
            e = self._class_embed(prev_class).view(-1, 1, A.size(2), A.size(3))
            h, c = self._rnn(th.cat((A, e), 1), (h, c))
            g = F.avg_pool2d(h, self._cnn.asize[-1]).squeeze()
            scores = self._clf(th.cat((g, e.view(e.size(0), -1)), 1))
            stop = self._stop_conv(h)
            stop = self._stop(stop.view(x.size(0), -1))

            if self.training and self._sample != 'greedy':
                if self._sample == 'stochastic':
                    prev_class = F.softmax(scores).multinomial(1).squeeze().detach()
                elif self._sample == 'oracle':
                    prev_class = samples[:, t]
            else:
                prev_class = scores.max(1)[1]

            path['scores'].append(scores.unsqueeze(1))
            path['stop'].append(stop)
            path['samples'].append(prev_class.unsqueeze(1))

        path['scores'] = th.cat(path['scores'], 1)
        path['stop'] = th.cat(path['stop'], 1)      # B x T
        path['samples'] = th.cat(path['samples'], 1)
        return path

    def get_preds(self, scores, stop):
        preds = scores.max(2)[1].data.long()
        stop = th.round(F.sigmoid(stop))

        stop_label = scores.size(2)
        mask = (stop.cumsum(1) > 0).data
        preds.masked_fill_(mask, stop_label)

        preds_onehot = util.onehot_sequence(preds, scores.size(2)+1)
        p = {'preds': preds, 'onehot': preds_onehot}
        return p


class RLModel(nn.Module):
    def __init__(self, cnn, num_classes, use_cuda, _):
        super(RLModel, self).__init__()
        self._cnn = cnn
        hidden_channels = self._cnn.asize[0]
        self._rnn = ConvLSTM(self._cnn.asize[0]+1, hidden_channels)

        embed_size = self._cnn.asize[1]*self._cnn.asize[2]
        self._clf = nn.Linear(self._cnn.asize[0]+embed_size, num_classes)
        self._class_embed = nn.Embedding(num_classes, embed_size)
        self._stop_conv = nn.Conv2d(self._cnn.asize[0], 1, kernel_size=3, padding=1)
        self._stop = nn.Linear(self._cnn.asize[1]*self._cnn.asize[2], 1)
        self._vf_conv = nn.Conv2d(self._cnn.asize[0], 1, kernel_size=3, padding=1)
        self._vf = nn.Linear(self._cnn.asize[1] * self._cnn.asize[2], 1)

    def forward(self, x, labels):
        path = defaultdict(list)
        A = self._cnn.activations(x)
        h = Variable(util.cuda_as(th.zeros(A.size()), A))
        c = Variable(util.cuda_as(th.zeros(A.size()), A))
        prev_class = Variable(util.cuda_as(th.zeros(x.size(0)), x).long())

        for t in range(labels.size(1)):
            e = self._class_embed(prev_class).view(-1, 1, A.size(2), A.size(3))
            h, c = self._rnn(th.cat((A, e), 1), (h, c))
            g = F.avg_pool2d(h, self._cnn.asize[-1]).squeeze()
            scores = self._clf(th.cat((g, e.view(e.size(0), -1)), 1))

            stop = self._stop_conv(h)
            stop = self._stop(stop.view(x.size(0), -1))
            v = self._vf_conv(h.detach())
            v = self._vf(v.view(x.size(0), -1))

            if self.training:
                prev_class = F.softmax(scores).multinomial(1).squeeze().detach()
            else:
                prev_class = scores.max(1)[1]

            path['scores'].append(scores.unsqueeze(1))
            path['stop'].append(stop)
            path['values'].append(v)
            path['samples'].append(prev_class.unsqueeze(1))

        path['scores'] = th.cat(path['scores'], 1)
        path['stop'] = th.cat(path['stop'], 1)      # B x T
        path['values'] = th.cat(path['values'], 1)
        path['samples'] = th.cat(path['samples'], 1)
        return path

    def get_preds(self, scores, stop):
        preds = scores.max(2)[1].data.long()
        stop = th.round(F.sigmoid(stop))

        stop_label = scores.size(2)
        mask = (stop.cumsum(1) > 0).data
        preds.masked_fill_(mask, stop_label)

        preds_onehot = util.onehot_sequence(preds, scores.size(2)+1)
        p = {'preds': preds, 'onehot': preds_onehot}
        return p

