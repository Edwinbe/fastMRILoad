# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
from typing import Callable, List, Optional, Tuple

import fastmri
import h5py
import numpy as np
import torch.utils.data


# -----------------------------------------------------------------------------
#                Single coil knee dataset (as used in MICCAI'20)
# -----------------------------------------------------------------------------
class MICCAI2020Data(torch.utils.data.Dataset):
    # This is the same as fastMRI singlecoil_knee, except we provide a custom test split
    # and also normalize images by the mean norm of the k-space over training data
    KSPACE_WIDTH = 368
    KSPACE_HEIGHT = 640
    START_PADDING = 166
    END_PADDING = 202
    CENTER_CROP_SIZE = 320

    def __init__(
        self,
        #数据根目录的路径
        root: pathlib.Path,
        #数据增强函数
        transform: Callable,
        #要加载的k-空间数据的列数
        num_cols: Optional[int] = None,
        #要加载的k-空间数据的数量
        num_volumes: Optional[int] = None,
        #如果指定了此参数，则仅加载每个卷的指定数量的随机切片
        num_rand_slices: Optional[int] = None,
        #自定义数据拆分的名称
        custom_split: Optional[str] = None,
    ):
        self.transform = transform
        self.examples: List[Tuple[pathlib.PurePath, int]] = []

        self.num_rand_slices = num_rand_slices
        self.rng = np.random.RandomState(1234)

        files = []
        #pathlib.Path(root).iterdir() 返回一个迭代器，允许你遍历指定目录下的所有项目（文件和子目录）-> 读取所有文件
        for fname in list(pathlib.Path(root).iterdir()):
            data = h5py.File(fname, "r")
            if num_cols is not None and data["kspace"].shape[2] != num_cols:
                continue
            files.append(fname)

        if custom_split is not None:
            split_info = []
            with open(f"activemri/data/splits/knee_singlecoil/{custom_split}.txt") as f:
                for line in f:
                    split_info.append(line.rsplit("\n")[0])
            files = [f for f in files if f.name in split_info]
        #随机打乱文件顺序并去num_volumes个卷
        if num_volumes is not None:
            self.rng.shuffle(files)
            files = files[:num_volumes]

        for volume_i, fname in enumerate(sorted(files)):
            data = h5py.File(fname, "r")
            kspace = data["kspace"]

            if num_rand_slices is None:
                num_slices = kspace.shape[0]
                self.examples += [(fname, slice_id) for slice_id in range(num_slices)]
            else:
                slice_ids = list(range(kspace.shape[0]))
                self.rng.seed(seed=volume_i)
                self.rng.shuffle(slice_ids)
                self.examples += [
                    (fname, slice_id) for slice_id in slice_ids[:num_rand_slices]
                ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_id = self.examples[i]
        with h5py.File(fname, "r") as data:
            #从文件中获取kspace并选择指定的切片
            kspace = data["kspace"][slice_id]
            #kspace.real -> kspace数据实部 kspace.imag -> kspace数据虚部
            kspace = torch.from_numpy(np.stack([kspace.real, kspace.imag], axis=-1))
            #fastmri.ifftshift 函数进行频率域数据的移动，通常用于k-空间数据的中心化操作
            kspace = fastmri.ifftshift(kspace, dim=(0, 1))
            #ifft 函数执行逆傅立叶变换，将k-空间数据转换为目标图像域（空间域）数据。normalized=False 表示不进行标准化
            target = torch.ifft(kspace, 2, normalized=False)
            #再次移动 ->保证中心的正确性
            target = fastmri.ifftshift(target, dim=(0, 1))
            # Normalize using mean of k-space in training data 归一化
            target /= 7.072103529760345e-07
            kspace /= 7.072103529760345e-07

            # Environment expects numpy arrays. The code above was used with an older
            # version of the environment to generate the results of the MICCAI'20 paper.
            # So, to keep this consistent with the version in the paper, we convert
            # the tensors back to numpy rather than changing the original code.
            kspace = kspace.numpy()
            target = target.numpy()
            return self.transform(
                kspace,
                torch.zeros(kspace.shape[1]),
                target,
                dict(data.attrs),
                fname.name,
                slice_id,
            )
