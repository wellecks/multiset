import argparse
from pprint import pprint

import numpy as np
import torch as th
import torch.optim as optim
from torch.autograd.variable import Variable
import datasets
import losses
import util
from util import cuda_as
from visualizer import Visualizer
from model import Model, RLModel
from cnn import CNN, ResNet

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', default=None)
parser.add_argument('--dataset-path', required=True)
parser.add_argument('--loss', required=True, choices=['multiset_loss', 'ce_loss', 'l1_loss', 'kl_loss', 'rl'])
parser.add_argument('--label-order', choices=['random', 'fixed_random', 'area', 'spatial'], default='random')
parser.add_argument('--randomize-batch-labels', action='store_true')

parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--max-objects', type=int, default=4)
parser.add_argument('--sample', choices=['greedy', 'oracle', 'stochastic'], default='greedy')

parser.add_argument('--save-dir', default='./checkpoints')
parser.add_argument('--log-base-dir', default='results')
parser.add_argument('--output-suffix', default='multiset')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save-every', type=int, default=1)
parser.add_argument('--print-every', type=int, default=100)
parser.add_argument('--validate-every', type=int, default=1)
parser.add_argument('--use-cuda', action='store_true')
parser.add_argument('--hidden-size', type=int, default=128)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dataset-size', type=int, default=70000)

parser = util.add_mutually_exclusive_group(parser, ['mnist-custom', 'coco-easy', 'coco-medium', 'voc'])
opts = util.setup(parser.parse_args())

np.random.seed(opts['seed'])
th.manual_seed(opts['seed'])
if opts['use_cuda']:
    th.cuda.manual_seed_all(opts['seed'])

# Load dataset
if opts['mnist_custom']:
    trainset, testset = datasets.load_mnist_multi(opts['dataset_path'] + '.npz',
                                                  opts['dataset_path'] + '_labels.npz',
                                                  opts['dataset_path'] + '_bbox.npz',
                                                  opts['max_objects'],
                                                  size=opts['dataset_size'],
                                                  label_order=opts['label_order'],
                                                  randomize_dataset=opts['randomize_batch_labels'])
    opts['in_channels'] = 1
    opts['num_classes'] = 10
    opts['img_size'] = (100, 100)
    cnn_ = CNN(opts['img_size'], opts['in_channels'], opts['num_classes'], opts['hidden_size'])

if opts['coco_easy'] or opts['coco_medium']:
    opts['img_size'] = (600, 600)
    if opts['coco_easy']:
        category_ids_file = 'resources/coco_category_ids__2object_938area_24classes.txt'
        image_ids_file = 'resources/coco_image_ids__2object_938area_24classes.txt'
        opts['num_classes'] = 24
        opts['max_objects'] = 2
    elif opts['coco_medium']:
        category_ids_file = 'resources/coco_category_ids__1-4object_938area_23classes.txt'
        image_ids_file = 'resources/coco_image_ids__1-4object_938area_23classes.txt'
        opts['num_classes'] = 23
        opts['max_objects'] = 5  # add "no object" label

    trainset = datasets.load_coco_filtered(opts['dataset_path'], category_ids_file, image_ids_file,
                                           max_objects=opts['max_objects'],
                                           label_order=opts['label_order'])
    testset = datasets.load_coco_filtered(opts['dataset_path'], category_ids_file, image_ids_file,
                                          dataset='valid',
                                          max_objects=opts['max_objects'],
                                          label_order=opts['label_order'])
    opts['in_channels'] = 3
    cnn_ = ResNet(opts['img_size'], opts['num_classes'], opts['in_channels'])


trainloader = th.utils.data.DataLoader(trainset, batch_size=opts['batch_size'], shuffle=True, num_workers=0)
testloader = th.utils.data.DataLoader(testset, batch_size=opts['batch_size'], shuffle=False, num_workers=0)

# Initialization
loss_fn = getattr(losses, opts['loss'])
model_class = RLModel if opts['loss'] == 'rl' else Model
model = model_class(cnn_, opts['num_classes'], opts['use_cuda'], opts['sample'])

# Load existing model (if applicable)
if opts['model_path']:
    model.load_state_dict(th.load(opts['model_path']))

opts['visualizer'] = Visualizer(opts['log_directory'], opts['output_suffix'])

if opts['use_cuda']:
    model.cuda()

# Train
print('Training...')
model.train()
optimizer = optim.Adam(list(model.parameters()), lr=opts['lr'])
step = 0
history = []
best = 0.0
for epoch in range(opts['num_epochs']):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        opts['timer'].reset()
        bbox = None if len(data) == 2 else data[2]
        inputs = cuda_as(Variable(data[0]), model)
        labels = cuda_as(Variable(data[1]), model).squeeze()
        opts['timer'].stamp('prepare_data')

        path = model.forward(inputs, labels)
        opts['timer'].stamp('model.forward')

        if opts['loss'] == 'rl':
            loss = loss_fn(path['scores'], labels, path['samples'], path['values'])
        else:
            loss = loss_fn(path['scores'], labels, path['samples'])
        loss += losses.stop_loss(path['stop'], labels, stop_class=opts['num_classes'])
        opts['timer'].stamp('loss')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        opts['timer'].stamp('backward')

        step += 1
        if i % opts['print_every'] == 0:
            preds = model.get_preds(path['scores'], path['stop'])
            metrics = util.compute_metrics(preds, labels)
            metrics['loss'] = loss.data.cpu().numpy()[0]
            opts['timer'].stamp('compute_metrics')

            util.log_tensorboard(metrics, step)
            print('Epoch %d [Batch %d]' % (epoch, i))
            pprint(metrics)
            print(opts['timer'].report(include_itrs=False, format_options={'itr_name_width': 30}))
            opts['timer'].stamp('logging', quick_print=True)

            # Visualize step entropies
            entropies = util.per_step_entropy(path['scores']).data.cpu().numpy().ravel()
            opts['visualizer'].step_entropies(entropies)
            opts['timer'].stamp('entropy visualization', quick_print=True)

    if epoch % opts['validate_every'] == 0:
        metrics = util.validate(testloader, model)
        util.log_tensorboard(metrics, step)
        history.append(metrics)
        util.save_history(history, opts['log_directory'])
        opts['timer'].stamp('validate', quick_print=True)

        is_best = metrics['valid_em'] > best
        best = metrics['valid_em'] if is_best else best
        util.save_checkpoint(model.state_dict(),
                             base_dir=opts['save_dir'],
                             prefix='%s' % opts['output_suffix'],
                             log=True,
                             is_best=is_best)
        opts['timer'].stamp('saving')
print('Finished Training')


