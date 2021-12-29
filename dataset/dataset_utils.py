"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
import numpy as np
from PIL import Image
from typing import Optional, Callable, Tuple, Any, List
from megengine.data import dataset
from dataset.download import download_and_extract_archive


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return np.array(img.convert("RGB"))


class ImageList(dataset.VisionDataset):
    """A generic Dataset class for image classification

    Args:
        root (str): Root directory of dataset
        classes (list[str]): The names of all the classes
        data_list_file (str): File to read the image list from.
        transform (callable, optional): A function/transform that  takes in an PIL image \
            and returns a transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `data_list_file`, each line has 2 values in the following format.
        ::
            source_dir/dog_xxx.png 0
            source_dir/cat_123.png 1
            target_dir/dog_xxy.png 0
            target_dir/cat_nsdf3.png 1

        The first value is the relative path of an image, and the second value is the label of the corresponding image.
        If your data_list_file has different formats, please over-ride :meth:`~ImageList.parse_data_file`.
    """

    def __init__(
        self,
        root: str,
        classes: List[str],
        data_list_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(
            root
        )  # , transform=transform, target_transform=target_transform)
        self.samples = self.parse_data_file(data_list_file)
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.loader = pil_loader
        self.data_list_file = data_list_file

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index
            return (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None and target is not None:
        #     target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.samples)

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """Parse file to data list

        Args:
            file_name (str): The path of data file
            return (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                split_line = line.split()
                target = split_line[-1]
                path = " ".join(split_line[:-1])
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)

    @classmethod
    def domains(cls):
        """All possible domain in this dataset"""
        raise NotImplementedError


def download(root: str, file_name: str, archive_name: str, url_link: str):
    """
    Download file from internet url link.

    Args:
        root (str) The directory to put downloaded files.
        file_name: (str) The name of the unzipped file.
        archive_name: (str) The name of archive(zipped file) downloaded.
        url_link: (str) The url link to download data.

    .. note::
        If `file_name` already exists under path `root`, then it is not downloaded again.
        Else `archive_name` will be downloaded from `url_link` and extracted to `file_name`.
    """
    if not os.path.exists(os.path.join(root, file_name)):
        print("Downloading {}".format(file_name))
        # if os.path.exists(os.path.join(root, archive_name)):
        #     os.remove(os.path.join(root, archive_name))
        try:
            download_and_extract_archive(
                url_link,
                download_root=root,
                filename=archive_name,
                remove_finished=False,
            )
        except Exception:
            print("Fail to download {} from url link {}".format(archive_name, url_link))
            print(
                "Please check you internet connection."
                "Simply trying again may be fine."
            )
            exit(0)


def check_exits(root: str, file_name: str):
    """Check whether `file_name` exists under directory `root`. """
    if not os.path.exists(os.path.join(root, file_name)):
        print("Dataset directory {} not found under {}".format(file_name, root))
        exit(-1)


def read_list_from_file(file_name: str) -> List[str]:
    """Read data from file and convert each line into an element in the list"""
    result = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            result.append(line.strip())
    return result
