from pathlib import Path
from typing import List, Optional, Union, Tuple

import h5py
try:
    import hdf5plugin
except ImportError:
    pass
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import IterableDataset

from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.genx_utils.sequence_base import SequenceBase, get_objframe_idx_2_repr_idx
from data.utils.augmentor import RandomSpatialAugmentorGenX
from data.utils.types import DataType, DatasetType, LoaderDataDictGenX
from utils.timers import TimerDummy as Timer


def _scalar_as_1d_array(scalar: Union[int, float]):
    return np.atleast_1d(scalar)


def _get_ev_repr_range_indices(indices: np.ndarray, max_len: int) -> List[Tuple[int, int]]:
    """
    インデックスの配列 `indices` と最大長 `max_len` を受け取り、連続する範囲の
    開始・終了インデックスのリストを返す関数です。
    """
    meta_indices_stop = np.flatnonzero(np.diff(indices) > max_len)
    meta_indices_start = np.concatenate((np.atleast_1d(0), meta_indices_stop + 1))
    meta_indices_stop = np.concatenate((meta_indices_stop, np.atleast_1d(len(indices) - 1)))
    out = list()
    for meta_idx_start, meta_idx_stop in zip(meta_indices_start, meta_indices_stop):
        idx_start = max(indices[meta_idx_start] - max_len + 1, 0)
        idx_stop = indices[meta_idx_stop] + 1
        out.append((idx_start, idx_stop))
    return out


class SequenceForIter(SequenceBase):
    """
    stream 用のシーケンス。SequenceBase で定義されている共通処理（イベント表現の読み込みやラベルの取得等）
    を引き継ぎ、シーケンスを一定間隔で区切ったサブシーケンスとして扱います。
    """
    def __init__(self,
                 path: Path,
                 ev_representation_name: str,
                 sequence_length: int,
                 dataset_type: DatasetType,
                 downsample_by_factor_2: bool,
                 range_indices: Optional[Tuple[int, int]] = None):
        super().__init__(path=path,
                         ev_representation_name=ev_representation_name,
                         sequence_length=sequence_length,
                         dataset_type=dataset_type,
                         downsample_by_factor_2=downsample_by_factor_2,
                         only_load_end_labels=False)
        # イベント表現ファイル内の総サンプル数を取得
        with h5py.File(str(self.ev_repr_file), 'r') as h5f:
            num_ev_repr = h5f['data'].shape[0]
        if range_indices is None:
            repr_idx_start = max(self.objframe_idx_2_repr_idx[0] - sequence_length + 1, 0)
            repr_idx_stop = num_ev_repr
        else:
            repr_idx_start, repr_idx_stop = range_indices

        # 最初のラベルに対応するタイムスタンプからシーケンス開始位置を設定
        min_start_repr_idx = max(self.objframe_idx_2_repr_idx[0] - sequence_length + 1, 0)
        assert 0 <= min_start_repr_idx <= repr_idx_start < repr_idx_stop <= num_ev_repr, \
            f'{min_start_repr_idx=}, {repr_idx_start=}, {repr_idx_stop=}, {num_ev_repr=}, {path=}'

        # シーケンスの開始・終了インデックスを等間隔に設定
        self.start_indices = list(range(repr_idx_start, repr_idx_stop, sequence_length))
        self.stop_indices = self.start_indices[1:] + [repr_idx_stop]
        self.length = len(self.start_indices)
        self._padding_representation = None

    @staticmethod
    def get_sequences_with_guaranteed_labels(
            path: Path,
            ev_representation_name: str,
            sequence_length: int,
            dataset_type: DatasetType,
            downsample_by_factor_2: bool) -> List['SequenceForIter']:
        """
        各サンプル内に必ずラベルが存在するシーケンスを生成するためのクラスメソッド。
        トレーニング時に必ずラベルが含まれるよう保証します。
        """
        objframe_idx_2_repr_idx = get_objframe_idx_2_repr_idx(
            path=path, ev_representation_name=ev_representation_name)
        range_indices_list = _get_ev_repr_range_indices(indices=objframe_idx_2_repr_idx, max_len=sequence_length)
        sequence_list = []
        for range_indices in range_indices_list:
            sequence_list.append(
                SequenceForIter(path=path,
                                ev_representation_name=ev_representation_name,
                                sequence_length=sequence_length,
                                dataset_type=dataset_type,
                                downsample_by_factor_2=downsample_by_factor_2,
                                range_indices=range_indices)
            )
        return sequence_list

    @property
    def padding_representation(self) -> torch.Tensor:
        if self._padding_representation is None:
            # 先頭のイベント表現を取得し、同じ形状のゼロテンソルをパディング用として設定
            ev_repr = self._get_event_repr_torch(start_idx=0, end_idx=1)[0]
            self._padding_representation = torch.zeros_like(ev_repr)
        return self._padding_representation

    def get_fully_padded_sample(self) -> LoaderDataDictGenX:
        """
        シーケンスの長さに満たない場合、全てパディングしたサンプルを返します。
        """
        is_first_sample = False
        is_padded_mask = [True] * self.seq_len
        ev_repr = [self.padding_representation] * self.seq_len
        labels = [None] * self.seq_len
        sparse_labels = SparselyBatchedObjectLabels(sparse_object_labels_batch=labels)
        out = {
            DataType.EV_REPR: ev_repr,
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.IS_FIRST_SAMPLE: is_first_sample,
            DataType.IS_PADDED_MASK: is_padded_mask,
        }
        return out

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> LoaderDataDictGenX:
        start_idx = self.start_indices[index]
        end_idx = self.stop_indices[index]
        sample_len = end_idx - start_idx
        assert self.seq_len >= sample_len > 0, f'{self.seq_len=}, {sample_len=}, {start_idx=}, {end_idx=}, ' \
                                               f'\n{self.start_indices=}\n{self.stop_indices=}'
        is_first_sample = (index == 0)
        is_padded_mask = [False] * sample_len

        with Timer(timer_name='read ev reprs'):
            ev_repr = self._get_event_repr_torch(start_idx=start_idx, end_idx=end_idx)
        assert len(ev_repr) == sample_len

        labels = []
        for repr_idx in range(start_idx, end_idx):
            labels.append(self._get_labels_from_repr_idx(repr_idx))
        assert len(labels) == len(ev_repr)

        # シーケンス長に満たない場合はパディングを実施
        if sample_len < self.seq_len:
            padding_len = self.seq_len - sample_len
            is_padded_mask.extend([True] * padding_len)
            ev_repr.extend([self.padding_representation] * padding_len)
            labels.extend([None] * padding_len)
        sparse_labels = SparselyBatchedObjectLabels(sparse_object_labels_batch=labels)

        out = {
            DataType.EV_REPR: ev_repr,
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.IS_FIRST_SAMPLE: is_first_sample,
            DataType.IS_PADDED_MASK: is_padded_mask,
        }
        return out


class RandAugmentIterDataset(IterableDataset):
    """
    torchdata の IterDataPipe の代替として、PyTorch 標準の IterableDataset を継承した
    ランダムな空間変換（Data Augmentation）を適用するためのクラスです。

    source_iterable には、SequenceForIter などから得られるサンプル（辞書型）を渡します。
    """
    def __init__(self, source_iterable, dataset_config: DictConfig):
        super().__init__()
        self.source_iterable = source_iterable

        resolution_hw = tuple(dataset_config.resolution_hw)
        assert len(resolution_hw) == 2, f"resolution_hw should be a tuple of length 2, got {resolution_hw}"
        ds_by_factor_2 = dataset_config.downsample_by_factor_2
        if ds_by_factor_2:
            resolution_hw = tuple(x // 2 for x in resolution_hw)

        augm_config = dataset_config.data_augmentation
        self.spatial_augmentor = RandomSpatialAugmentorGenX(
            dataset_hw=resolution_hw,
            automatic_randomization=False,
            augm_config=augm_config.stream)

    def __iter__(self):
        # 各イテレーション開始時にランダムな変換パラメータを生成
        self.spatial_augmentor.randomize_augmentation()
        for sample in self.source_iterable:
            yield self.spatial_augmentor(sample)
