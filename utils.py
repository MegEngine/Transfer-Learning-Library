import bisect
import megengine.data.transform as T
import megengine.functional as F
from megengine.data.dataset import Dataset
from megengine.data import DataLoader
from office31 import Office31


def get_train_transform(
    resizing='default',
    random_horizontal_flip=True,
    random_color_jitter=False,
    resize_size=224,
    norm_mean=(0.485, 0.456, 0.406),
    norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    if resizing == 'default':
        transform = T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = T.Resize(resize_size)
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(T.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.Normalize(mean=norm_mean, std=norm_std),
        T.ToMode('CHW')
    ])
    return T.Compose(transforms)


def get_val_transform(
    resizing='default',
    resize_size=224,
    norm_mean=(0.485, 0.456, 0.406),
    norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = T.Resize(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.Normalize(mean=norm_mean, std=norm_std),
        T.ToMode('CHW')
    ])


class ConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets) -> None:
        super(ConcatDataset, self).__init__()
        # Cannot verify that datasets is Sized
        assert len(datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name != 'Office31':
        raise NotImplementedError
    def concat_dataset(tasks, **kwargs):
        return ConcatDataset([Office31(task=task, **kwargs) for task in tasks])

    train_source_dataset = concat_dataset(root=root, tasks=source, download=True, transform=train_source_transform)
    train_target_dataset = concat_dataset(root=root, tasks=target, download=True, transform=train_target_transform)
    val_dataset = concat_dataset(root=root, tasks=target, download=True, transform=val_transform)
    test_dataset = val_dataset

    class_names = train_source_dataset.datasets[0].classes
    num_classes = len(class_names)

    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


def accuracy(output, target, topk=1):
    batch_size = target.shape[0]
    _, pred = F.topk(output, k=1, descending=True)
    pred = F.flatten(pred)
    correct = sum(pred == target) / batch_size
    return correct * 100
