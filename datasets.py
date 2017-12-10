"""Utilities for loading datasets."""
import os
import cv2
import numpy as np
import torch as th
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

def load_mnist_multi(dataset_path, labels_path, bbox_path, max_objects, size=70000, train_pct=0.9,
                     label_order='spatial', randomize_dataset=False, seed=42):
    """Load MNIST Multi into `BBoxDataset`s (train and validation).
    Each example's labels are padded with an extra '10' class so that every example has
    `max_objects` labels.
    E.g. Image has digits 0, 4, 6 and max_objects=3, then labels = [0, 4, 6]
         Image has digits 1, 8    and max_objects=3, then labels = [1, 8, 10]

    `label_order` is 'random' | 'fixed_random' | 'area' | 'spatial' (default)
    `randomize_dataset` randomizes the label order whenever an example is accessed
        (e.g. whenever a mini-batch is sampled) from the output `BBoxDataset`.
        An evaluation of label order invariance should use randomize_dataset=True.

    To generate a test set, use `train_pct`=1.0.
    """
    dataset_np = np.load(dataset_path).items()[0][1][:size]
    labels_np = np.load(labels_path).items()[0][1][:size]
    bbox_np = np.load(bbox_path).items()[0][1][:size]

    # Label ordering.
    if label_order == 'fixed_random':
        rng = np.random.RandomState(seed)
        fixed_random_order = list(rng.permutation(10))
        for i in xrange(labels_np.shape[0]):
            labels_np[i] = np.array(sorted(list(labels_np[i]),
                                           key=lambda x: fixed_random_order.index(x)))
            bbox_np[i] = np.array(sorted(list(bbox_np[i]),
                                         key=lambda x: fixed_random_order.index(x)))
        pass
    elif label_order == 'area':
        for i in xrange(labels_np.shape[0]):
            areas = [np.prod(bbox[2:]-bbox[:2]) for bbox in bbox_np[i]]
            triples = [(labels_np[i][j], bbox_np[i][j], area) for j, area in enumerate(areas)]
            sorted_triples = sorted(triples, key=lambda x: x[2], reverse=True)
            labels_np[i] = np.array([x[0] for x in sorted_triples])
            bbox_np[i] = np.array([x[1] for x in sorted_triples])
        pass
    elif label_order == 'random':
        for i in xrange(labels_np.shape[0]):
            idxs = np.random.permutation(len(labels_np[i]))
            labels_np[i] = labels_np[i][idxs]
            bbox_np[i] = bbox_np[i][idxs]
    else:  # MNIST Multi labels are ordered spatially by default
        pass

    # Pad with extra class.
    labels_np = np.array([np.concatenate([ls, [10] * (max_objects - ls.shape[0])])
                          for ls in labels_np]).astype(int)

    dataset_t, labels_t = _np_to_t(dataset_np, labels_np)
    train_size = int(size * train_pct)
    X_train_t, y_train_t = dataset_t[:train_size], labels_t[:train_size]

    bbox_np = np.array([np.vstack([bbox, np.zeros((max_objects - bbox.shape[0], bbox.shape[1])) - 1])
                        for bbox in bbox_np])
    bbox_train = bbox_np[:train_size]
    trainset = BBoxDataset(X_train_t, y_train_t, bbox_train,
                           stop_class=10, randomize=randomize_dataset)

    validset = None
    if train_pct != 1.0:
        bbox_test = bbox_np[train_size:]
        X_valid_t, y_valid_t = dataset_t[train_size:], labels_t[train_size:]
        validset = BBoxDataset(X_valid_t, y_valid_t, bbox_test,
                               stop_class=10, randomize=randomize_dataset)

    return trainset, validset


def _np_to_t(x, y):
    """Convert numpy array `x` and numpy integer array `y` into torch Tensors"""
    x = th.from_numpy(x).float()
    x.unsqueeze_(1)
    y = th.LongTensor(y)
    return x, y


def resize_and_pad(pic, rows=600, cols=600, num_channels=3):
    """Ref: https://github.com/marcellacornia/sam/blob/master/utilities.py"""
    pic = np.asarray(pic)
    pic_padded = np.zeros((rows, cols, num_channels), dtype=np.uint8)
    if num_channels == 1:
        pic_padded = np.zeros((rows, cols), dtype=np.uint8)
    original_shape = pic.shape
    row_rate = original_shape[0] / rows
    col_rate = original_shape[1] / cols
    if row_rate > col_rate:
        new_cols = (original_shape[1] * rows) // original_shape[0]
        pic = cv2.resize(pic, (new_cols, rows))
        if new_cols > cols:
            new_cols = cols
        pic_padded[:, (cols-new_cols)//2:(cols-new_cols)//2+new_cols] = pic
    else:
        new_rows = (original_shape[0] * cols) // original_shape[1]
        if new_rows > rows:
            new_rows = rows
        pic = cv2.resize(pic, (cols, new_rows))
        pic_padded[(rows-new_rows)//2:(rows-new_rows)//2+new_rows, :] = pic

    return pic_padded


def load_coco(root, dataset='train', randomize=True):
    """Assume folders `annotations` `train2014` and `val2014` folders are in root dir."""
    transform = transforms.ToTensor()
    img_path = os.path.join(root, '%s2014' % dataset)
    ann_file = os.path.join(root, 'annotations', 'instances_%s2014.json' % dataset)
    ds = CocoDetection(img_path, ann_file, transform=transform, randomize=randomize)
    return ds


def load_coco_filtered(root, category_ids_file, image_ids_file, dataset='train',
                       max_objects=4, label_order='random'):
    # dataset='train' is training set, dataset='val' is _test_ set, anything else is validation set
    d = 'val' if dataset == 'val' else 'train'
    img_path = os.path.join(root, '%s2014' % d)
    ann_file = os.path.join(root, 'annotations', 'instances_%s2014.json' % d)
    min_area = int(category_ids_file.split('_')[-2][:-4])
    category_ids = map(int, list(np.loadtxt(category_ids_file)))
    image_ids = map(int, list(np.loadtxt(image_ids_file)))
    if dataset == 'train':
        image_ids = image_ids[:int(len(image_ids)*0.85)]
    elif dataset == 'val':  # val is actually our test set, it refers to the COCO val set
        image_ids = image_ids
    else:
        image_ids = image_ids[int(len(image_ids)*0.85):]
    annotation_filter = lambda ann: ann['area'] > min_area and ann['category_id'] in category_ids
    transform = transforms.Compose([transforms.Lambda(lambda x: resize_and_pad(x)),
                                    transforms.ToTensor()])
    ds = CocoDetection(img_path, ann_file, max_objects,
                       image_ids=image_ids,
                       category_ids=category_ids,
                       annotation_filter=annotation_filter,
                       transform=transform,
                       label_order=label_order)
    return ds


class CocoDetection(Dataset):
    def __init__(self, img_path, ann_file, max_objects,
                 image_ids=None,
                 category_ids=None,
                 annotation_filter=lambda x: True,
                 transform=None,
                 label_order='random'):
        """`label_order` is 'random' | 'fixed_random' | 'area' | 'spatial'

        label_order='random' randomizes the label order _every time_ an example is accessed.
        An evaluation of label order invariance should use label_order='random'.
        """
        from pycocotools.coco import COCO
        self.img_path = img_path
        self.max_objects = max_objects
        self.coco = COCO(ann_file)
        self.image_ids = image_ids if image_ids is not None else list(self.coco.imgs.keys())
        self.category_ids = category_ids if category_ids is not None else self.coco.getCatIds()
        category_names = [c['name'] for c in self.coco.loadCats(self.category_ids)]
        self.category_id_to_label = {c: i for i, c in enumerate(category_ids)}
        self.label_to_category = {i: cn for i, cn in enumerate(category_names)}
        self.transform = transform
        self.ann_filter = annotation_filter
        self.label_order = label_order

        # fixed random ordering
        rng = np.random.RandomState(42)
        self.fixed_random_order = list(rng.permutation(len(category_ids)))

    def __getitem__(self, index):
        coco = self.coco
        image_id = self.image_ids[index]
        ann_ids = coco.getAnnIds(imgIds=image_id)
        target = coco.loadAnns(ann_ids)

        # spatial location ordering
        if self.label_order == 'spatial':
            target = sorted(target, key=lambda x: x['bbox'][:2])

        # fixed random ordering
        if self.label_order == 'fixed_random':
            target = filter(self.ann_filter, target)
            target = sorted(target, key=lambda x: self.fixed_random_order.index(
                self.category_id_to_label[x['category_id']]))

        # area ordering
        if self.label_order == 'area':
            target = sorted(target, key=lambda x: x['area'], reverse=True)

        label_np = np.array([self.category_id_to_label[ann['category_id']]
                             for ann in target if self.ann_filter(ann)])
        bbox_np = np.array([ann['bbox'] for ann in target if self.ann_filter(ann)]).astype(np.float)

        path = coco.loadImgs(image_id)[0]['file_name']

        img = Image.open(os.path.join(self.img_path, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # random ordering
        if self.label_order == 'random':
            idxs = np.random.permutation(label_np.shape[0])
            label_np = label_np[idxs]
            bbox_np = bbox_np[idxs]

        # Pad labels, bboxes to length `max_objects` with stop class labels and 0 bboxes, resp.
        label_np = np.array(
            [np.concatenate([label_np, [len(self.category_ids)] * (self.max_objects - label_np.shape[0])])]).astype(int)
        label_t = th.from_numpy(label_np).squeeze(1)
        bbox_np = np.vstack((bbox_np, np.zeros((label_np.shape[1] - bbox_np.shape[0], 4))))
        return img, label_t, bbox_np

    def __len__(self):
        return len(self.image_ids)


class BBoxDataset(Dataset):
    def __init__(self, dataset_t, labels_t, bbox_np,
                 randomize=False, input_transform=None, target_transform=None, stop_class=10):
        """Each item is (image, label, bbox) where bbox is a 4-tuple (x_nw, y_nw, x_se, y_se)."""
        self.dataset_t = dataset_t
        self.labels_t = labels_t
        self.bbox_np = bbox_np
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.stop_class = stop_class
        self.randomize = randomize

    def __getitem__(self, index):
        image = self.dataset_t[index]
        label = self.labels_t[index]
        if self.randomize:
            k = (label != self.stop_class).sum()
            if k < label.size(0):
                label = th.cat((label[th.randperm(k)], label[k:]))
            else:
                label = label[th.randperm(k)]

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, self.bbox_np[index]

    def __len__(self):
        return self.dataset_t.size(0)