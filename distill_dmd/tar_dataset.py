import io
import os
import pickle
import tarfile
from functools import lru_cache

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from util.crop import center_crop_arr


def build_flat_index(tar_path: str, index_path: str):
    """Build a flat offset index for original ImageNet train tar files.

    The official `ILSVRC2012_img_train.tar` contains one tar per class. We index
    the nested JPEG byte ranges once, then each worker can seek directly into the
    outer tar without extracting millions of files.
    """
    if os.path.exists(index_path):
        print(f"Using existing tar index: {index_path}")
        with open(index_path, "rb") as handle:
            return pickle.load(handle)

    entries = []
    class_names = set()
    with tarfile.open(tar_path, "r:") as outer:
        for class_tar in outer.getmembers():
            if not class_tar.isfile() or not class_tar.name.endswith(".tar"):
                continue
            outer_offset = class_tar.offset_data
            class_fileobj = outer.extractfile(class_tar)
            with tarfile.open(fileobj=class_fileobj, mode="r:") as inner:
                for member in inner.getmembers():
                    if not member.isfile():
                        continue
                    class_name = member.name.split("_", 1)[0]
                    class_names.add(class_name)
                    entries.append((outer_offset + member.offset_data, member.size, class_name))

    class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}
    flat_entries = [(offset, size, class_to_idx[class_name]) for offset, size, class_name in entries]

    index_dir = os.path.dirname(index_path)
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)
    with open(index_path, "wb") as handle:
        pickle.dump(flat_entries, handle)
    print(f"Built tar index with {len(flat_entries)} images: {index_path}")
    return flat_entries


class ImageNetTarDataset(Dataset):
    """Read ImageNet train images directly from `ILSVRC2012_img_train.tar`."""

    def __init__(self, tar_path: str, transform=None, index_path: str = ""):
        self.tar_path = tar_path
        self.transform = transform
        self.index_path = index_path or tar_path + ".index"
        self.entries = build_flat_index(tar_path, self.index_path)
        self.tar_handle = None

    def __len__(self):
        return len(self.entries)

    @lru_cache(maxsize=16)
    def _get_image(self, index):
        if self.tar_handle is None:
            self.tar_handle = open(self.tar_path, "rb")
        offset, size, label = self.entries[index]
        self.tar_handle.seek(offset)
        data = self.tar_handle.read(size)
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return image, label

    def __getitem__(self, index):
        image, label = self._get_image(index)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class MarTrainTransform:
    """MAR train augmentation without requiring an extracted ImageFolder."""

    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, pil_image):
        pil_image = center_crop_arr(pil_image, self.image_size)
        arr = np.array(pil_image)
        if torch.rand(1).item() < 0.5:
            arr = arr[:, ::-1, :]
        arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
        tensor = torch.from_numpy(arr).float().div_(255.0)
        tensor = tensor.sub_(0.5).div_(0.5)
        return tensor
