# dataset_streaming.py
from functools import partialmethod
from pathlib import Path
from typing import List, Union

from omegaconf import DictConfig
from tqdm import tqdm
import torch

# もともとの SequenceForIter はそのまま利用する前提です。
from data.genx_utils.sequence_stream import SequenceForIter
# torchdata 由来の ConcatStreamingDataPipe / ShardedStreamingDataPipe の代わりに、
# PyTorch 標準の IterableDataset を継承した新しいクラスを利用します。
from data.utils.stream_concat_datapipe import ConcatStreamingDataset
from data.utils.stream_sharded_datapipe import ShardedStreamingDataset
from data.utils.types import DatasetMode, DatasetType

# ─────────────────────────────────────────────
# augmentation 用の callable クラス
class RandAugmentFunction:
    def __init__(self, dataset_config: DictConfig):
        resolution_hw = tuple(dataset_config.resolution_hw)
        if dataset_config.downsample_by_factor_2:
            resolution_hw = tuple(x // 2 for x in resolution_hw)
        augm_config = dataset_config.data_augmentation
        # 必要な場合は RandomSpatialAugmentorGenX の実装に合わせて import パスを調整してください。
        from data.utils.augmentor import RandomSpatialAugmentorGenX
        self.augmentor = RandomSpatialAugmentorGenX(
            dataset_hw=resolution_hw,
            automatic_randomization=False,
            augm_config=augm_config.stream)
    def __call__(self, sample):
        self.augmentor.randomize_augmentation()
        return self.augmentor(sample)

# ─────────────────────────────────────────────
def build_streaming_dataset(dataset_mode: DatasetMode, dataset_config: DictConfig, batch_size: int, num_workers: int) -> Union[ConcatStreamingDataset, ShardedStreamingDataset]:
    dataset_path = Path(dataset_config.path)
    assert dataset_path.is_dir(), f'{str(dataset_path)}'

    mode2str = {DatasetMode.TRAIN: 'train',
                DatasetMode.VALIDATION: 'val',
                DatasetMode.TESTING: 'test'}

    split_path = dataset_path / mode2str[dataset_mode]
    assert split_path.is_dir()
    datasets = []
    num_full_sequences = 0
    num_splits = 0
    num_split_sequences = 0
    guarantee_labels = dataset_mode == DatasetMode.TRAIN
    for entry in tqdm(list(split_path.iterdir()), desc=f'creating streaming {mode2str[dataset_mode]} datasets'):
        new_datasets = get_sequences(path=entry, dataset_config=dataset_config, guarantee_labels=guarantee_labels)
        if len(new_datasets) == 1:
            num_full_sequences += 1
        else:
            num_splits += 1
            num_split_sequences += len(new_datasets)
        datasets.extend(new_datasets)
    print(f'num_full_sequences={num_full_sequences}\nnum_splits={num_splits}\nnum_split_sequences={num_split_sequences}')

    if dataset_mode == DatasetMode.TRAIN:
        return build_streaming_train_dataset(datasets=datasets, dataset_config=dataset_config, batch_size=batch_size, num_workers=num_workers)
    elif dataset_mode in (DatasetMode.VALIDATION, DatasetMode.TESTING):
        return build_streaming_evaluation_dataset(datasets=datasets, batch_size=batch_size)
    else:
        raise NotImplementedError

def get_sequences(path: Path, dataset_config: DictConfig, guarantee_labels: bool) -> List[SequenceForIter]:
    assert path.is_dir()

    # extract settings from config
    sequence_length = dataset_config.sequence_length
    ev_representation_name = dataset_config.ev_repr_name
    downsample_by_factor_2 = dataset_config.downsample_by_factor_2
    if dataset_config.name == 'gen1':
        dataset_type = DatasetType.GEN1
    elif dataset_config.name == 'gen4':
        dataset_type = DatasetType.GEN4
    elif dataset_config.name == 'VGA':
        dataset_type = DatasetType.VGA
    else:
        raise NotImplementedError

    if guarantee_labels:
        return SequenceForIter.get_sequences_with_guaranteed_labels(
            path=path,
            ev_representation_name=ev_representation_name,
            sequence_length=sequence_length,
            dataset_type=dataset_type,
            downsample_by_factor_2=downsample_by_factor_2)
    return [SequenceForIter(
        path=path,
        ev_representation_name=ev_representation_name,
        sequence_length=sequence_length,
        dataset_type=dataset_type,
        downsample_by_factor_2=downsample_by_factor_2)]

def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return NewCls

def build_streaming_train_dataset(datasets: List[SequenceForIter],
                                  dataset_config: DictConfig,
                                  batch_size: int,
                                  num_workers: int) -> ConcatStreamingDataset:
    assert len(datasets) > 0
    augmentation_function = RandAugmentFunction(dataset_config)
    streaming_dataset = ConcatStreamingDataset(dataset_list=datasets,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               augmentation_pipeline=augmentation_function,
                                               print_seed_debug=False)
    return streaming_dataset

def build_streaming_evaluation_dataset(datasets: List[SequenceForIter],
                                       batch_size: int) -> ShardedStreamingDataset:
    assert len(datasets) > 0
    fill_value = datasets[0].get_fully_padded_sample()
    streaming_dataset = ShardedStreamingDataset(dataset_list=datasets, batch_size=batch_size, fill_value=fill_value)
    return streaming_dataset
