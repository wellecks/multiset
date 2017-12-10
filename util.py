import os
import shutil
import datetime
import inspect
import gtimer as gt
import pickle

import numpy as np
import torch as th
import torch.nn as nn
from pprint import pprint
from torch.autograd import Variable
import tensorboard_logger as logger
import torch.nn.functional as F


def mkdir_p(path, log=True):
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    if log:
        print('Created directory %s' % path)

def date_filename(base_dir='./'):
    dt = datetime.datetime.now()
    return os.path.join(base_dir, '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second))

def add_mutually_exclusive_group(parser, choice_names):
    group = parser.add_mutually_exclusive_group(required=True)
    for name in choice_names:
        group.add_argument('--%s' % name, action='store_true')
    return parser

def load_checkpoint(base_dir='./', prefix='model', log=False):
    filename = os.path.join(base_dir, '%s_checkpoint.pth.tar' % prefix)
    checkpoint = th.load(filename)
    if log:
        print('Loaded checkpoint %s' % filename)
    return checkpoint

def save_checkpoint(state, base_dir='./', prefix='model', log=False, is_best=False):
    filename = os.path.join(base_dir, '%s_checkpoint.pth.tar' % prefix)
    th.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(base_dir, '%s_best.pth.tar' % prefix))
    if log:
        print('Saved checkpoint %s' % filename)

def save_history(history, base_dir):
    filename = os.path.join(base_dir, 'history.pkl')
    with open(filename, 'w') as f:
        pickle.dump(history, f)
    print('Saved history %s' % filename)

def save_arg_dict(d, base_dir='./', filename='settings.txt', log=True):
    def _format_value(v):
        if isinstance(v, float):
            return '%.4f' % v
        elif isinstance(v, int):
            return '%d' % v
        else:
            return '%s' % str(v)

    with open(os.path.join(base_dir, filename), 'w') as f:
        for k, v in d.iteritems():
            f.write('%s\t%s\n' % (k, _format_value(v)))
    if log:
        print('Saved settings to %s' % os.path.join(base_dir, filename))

def setup_tensorboard(opts):
    """Creates a logging directory and configures a tensorboard logger."""
    log_directory = date_filename(opts['log_base_dir']) + opts['output_suffix']
    mkdir_p(log_directory)
    try:
        logger.configure(log_directory)
    except ValueError:
        pass
    return log_directory

def log_tensorboard(values_dict, step):
    for k, v in values_dict.iteritems():
        logger.log_value(k, v, step)

def setup(args, log=True):
    """Perform boilerplate experiment setup steps, returning a dictionary of config options."""
    opts = args.__dict__.copy()
    np.random.seed(opts['seed'])
    mkdir_p(opts['save_dir'], log=log)
    log_directory = setup_tensorboard(opts)
    save_arg_dict(opts, log_directory, log=log)
    opts['log_directory'] = log_directory
    opts['logger'] = logger
    opts['timer'] = gt
    if log:
        pprint(opts)
    return opts

def cuda_as(x, ref):
    """Set CUDA status of `x` according to `ref`'s CUDA status."""
    if isinstance(ref, bool):
        cuda = ref
    elif isinstance(ref, nn.Module):
        cuda = ref.parameters().next().is_cuda
    elif inspect.ismethod(ref):
        cuda = ref.im_self.parameters().next().is_cuda
    else:
        cuda = ref.is_cuda
    return x.cuda() if cuda else x.cpu()

def sample_permutation(labels, stop_class):
    """Sample a permutation of each variable-length labels multiset
    (a row of `labels`, padded to a maximum size with `stop_class`)."""
    mask = (labels != stop_class)
    numel = mask.sum(1).data.int()
    permutation_indices = []
    for i in range(len(numel)):
        perm = cuda_as(th.randperm(numel[i]), labels)
        if len(perm) != labels.size(1):
            # Pad STOP indices with zeros (can be any index since it will be masked)
            perm = th.cat((perm, cuda_as(th.zeros(labels.size(1) - len(perm)).long(), labels)))
        permutation_indices.append(perm.unsqueeze(0))
    permutation_indices = th.stack(permutation_indices, 1).squeeze()

    samples = th.gather(labels, 1, permutation_indices)
    return samples

def infer_shape(in_size, operation, batch_size=1):
    tensor = Variable(cuda_as(th.ones(batch_size, *in_size), operation))
    f = operation(tensor)
    return f.size()[1:]

def onehot(labels, D):
    # labels (N, 1)
    labels_ = labels.cpu()
    N = labels_.size()[0]
    result = th.zeros(N, D)
    result.scatter_(1, labels_, 1)
    result = cuda_as(result, labels)
    return result

def onehot_sequence(labels, D):
    # labels (N, T)
    N, T = labels.size()
    onehot_labels = []
    for l in th.chunk(labels, T, 1):
        onehot_label = onehot(l, D)
        onehot_label = th.unsqueeze(onehot_label, 1)
        onehot_labels.append(onehot_label)
    onehot_labels = th.cat(onehot_labels, 1)
    return onehot_labels

def perclass_precision(pred, actual):
    prec = (np.minimum(pred, actual) / np.maximum(pred, 1.0))
    prec_ = np.ma.array(prec, mask=actual == 0)
    prec = prec_.mean(0)
    return prec

def perclass_recall(pred, actual):
    rec = (np.minimum(pred, actual) / np.maximum(actual, 1.0))
    rec_ = np.ma.array(rec, mask=actual == 0)
    rec = rec_.mean(0)
    return rec

def f1_score(prec, rec):
    prec = prec.compressed()
    rec = rec.compressed()
    denom = prec + rec
    num = 2.0 * (prec * rec)
    f1 = np.array((num[denom != 0] / denom[denom != 0])).sum() / float(len(rec))
    return f1

def exact_match(pred, actual):
    match_digits = (pred == actual).sum(axis=1)
    em = (match_digits == pred.shape[1]).sum()
    return em

def pct_match(pred, actual):
    return (pred == actual).sum() / float(actual.numel())

def per_step_entropy(scores):
    """Averaged across batch dimension"""
    scores = Variable(scores.data, volatile=True)
    entropies = []
    for t in range(scores.size(1)):
        log_ps = F.log_softmax(scores[:, t, :])
        h = categorical_entropy(log_ps)
        entropies.append(h.mean())
    entropies = th.cat(entropies)
    return entropies

def categorical_entropy(log_ps):
    h = -th.sum(th.exp(log_ps)*log_ps, 1)
    return h

def compute_metrics(preds, labels):
    preds_onehot = preds['onehot'].sum(1).squeeze(1).cpu().numpy()
    labels_onehot = onehot_sequence(labels.data, preds_onehot.shape[1]).sum(1)
    labels_onehot = labels_onehot.cpu().numpy()

    prec = perclass_precision(preds_onehot, labels_onehot)
    rec = perclass_recall(preds_onehot, labels_onehot)
    em = exact_match(preds_onehot, labels_onehot) / float(labels.size(0))
    f1 = f1_score(prec, rec)
    metrics = dict(prec=prec.mean(), rec=rec.mean(), em=em, f1=f1)
    return metrics

def validate(dataloader, model):
    model.eval()
    lst = []
    for i, data in enumerate(dataloader):
        inputs = cuda_as(Variable(data[0], volatile=True), model)
        labels = cuda_as(Variable(data[1], volatile=True), model).squeeze()
        path = model.forward(inputs, labels)
        preds = model.get_preds(path['scores'], path['stop'])
        lst.append(compute_metrics(preds, labels))
    metrics = {}
    for key in lst[0]:
        metrics['valid_' + key] = np.mean([_[key] for _ in lst])
    model.train()
    return metrics
