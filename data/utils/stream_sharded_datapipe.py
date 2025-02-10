# stream_sharded_datapipe.py
from typing import Any, List, Optional, Iterator, Tuple
import itertools
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

class ShardedStreamingDataset(IterableDataset):
    """
    VALIDATION/TESTING 用のデータセットです。
    全体のシーケンス群からワーカー毎に担当シーケンスをピラミッド順に割り当て，
    各グループ内で連結してバッチサンプルを生成します。
    
    データが不足する場合は、fill_value で補完します。
    """
    def __init__(self, dataset_list: List[Any], batch_size: int, fill_value: Optional[Any] = None):
        super().__init__()
        assert batch_size > 0
        # 各シーケンスの長さ（len(ds)）で降順ソート
        self.dataset_list = sorted(dataset_list, key=lambda x: len(x), reverse=True)
        self.batch_size = batch_size
        self.fill_value = fill_value

    @staticmethod
    def yield_pyramid_indices(start_idx: int, end_idx: int) -> Iterator[int]:
        while True:
            for idx in range(start_idx, end_idx):
                yield idx
            for idx in range(end_idx - 1, start_idx - 1, -1):
                yield idx

    @classmethod
    def assign_datasets_to_worker(cls,
                                  sorted_dataset_list: List[Any],
                                  total_num_workers: int,
                                  global_worker_id: int) -> List[Any]:
        num_datasets = len(sorted_dataset_list)
        assert num_datasets >= total_num_workers > global_worker_id, f'{num_datasets=}, {total_num_workers=}, {global_worker_id=}'
        assigned = []
        pyramid_gen = cls.yield_pyramid_indices(0, total_num_workers)
        for ds in sorted_dataset_list:
            worker_id = next(pyramid_gen)
            if worker_id == global_worker_id:
                assigned.append(ds)
        assert len(sorted_dataset_list) > 0
        return assigned

    def get_zipped_stream_from_worker_datasets(self, dataset_list: List[Any], batch_size: int) -> Iterator[Tuple]:
        num_datasets = len(dataset_list)
        assert num_datasets > 0
        assert batch_size > 0
        print(f"num_datasets: {num_datasets}, batch_size: {batch_size}")
        assert num_datasets >= batch_size, "各ワーカーが少なくとも batch_size 個のシーケンスを担当する必要があります。ワーカー数を減らしてください。"
        # 長い順に再ソート
        dataset_list = sorted(dataset_list, key=lambda x: len(x), reverse=True)
        grouped = [[] for _ in range(batch_size)]
        batch_id_generator = self.yield_pyramid_indices(0, batch_size)
        for ds in dataset_list:
            batch_idx = next(batch_id_generator)
            grouped[batch_idx].append(ds)
        # 各グループ内のシーケンスを連結する無限イテレータを生成
        def concat_iterator(ds_list: List[Any]) -> Iterator[Any]:
            while True:
                for ds in ds_list:
                    for sample in ds:
                        yield sample
        iterators = []
        for group in grouped:
            if not group:
                iterators.append(itertools.repeat(self.fill_value))
            else:
                iterators.append(concat_iterator(group))
        return zip(*iterators)

    def __iter__(self) -> Iterator[Tuple[Tuple[Any, ...], int]]:
        worker_info = torch.utils.data.get_worker_info()
        local_worker_id = 0 if worker_info is None else worker_info.id
        local_num_workers = 1 if worker_info is None else worker_info.num_workers
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            global_rank = dist.get_rank()
        else:
            world_size = 1
            global_rank = 0
        total_num_workers = local_num_workers * world_size
        global_worker_id = global_rank * local_num_workers + local_worker_id

        local_datasets = self.assign_datasets_to_worker(sorted_dataset_list=self.dataset_list,
                                                        total_num_workers=total_num_workers,
                                                        global_worker_id=global_worker_id)
        zipped_stream = self.get_zipped_stream_from_worker_datasets(dataset_list=local_datasets,
                                                                     batch_size=self.batch_size)
        def worker_id_iterator(worker_id: int) -> Iterator[int]:
            while True:
                yield worker_id
        worker_id_iter = worker_id_iterator(local_worker_id)
        for batch, wid in zip(zipped_stream, worker_id_iter):
            yield (batch, wid)
