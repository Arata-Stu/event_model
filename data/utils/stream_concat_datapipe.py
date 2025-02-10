# stream_concat_datapipe.py
from typing import Any, Iterator, List, Optional, Tuple
import itertools
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

class ConcatStreamingDataset(IterableDataset):
    """
    このデータセットはバッチおよびワーカー単位でランダムなストリームの連結を行うことで、
    シャーディングの問題を回避します。
    
    Pros:
      - 各バッチに必ず有効なサンプルが含まれるため、バッチサイズは常に一定。
    Cons:
      - 同一バッチ内に重複サンプルが含まれる可能性がある（ただし、データ拡張により変化する）。
      - エポック内でデータセットを繰り返すため、検証・テストには不向き。
    
    ※ TRAIN 用として推奨。
    """
    def __init__(self,
                 dataset_list: List[Any],
                 batch_size: int,
                 num_workers: int,
                 augmentation_pipeline: Optional[Any] = None,
                 print_seed_debug: bool = False):
        super().__init__()
        assert batch_size > 0
        self.dataset_list = dataset_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        # augmentation_pipeline は callable として各サンプルに適用
        self.augmentation_pipeline = augmentation_pipeline if augmentation_pipeline is not None else (lambda x: x)
        self.print_seed_debug = print_seed_debug

    def random_shuffle_list(self, data: List[Any]) -> List[Any]:
        indices = torch.randperm(len(data)).tolist()
        return [data[i] for i in indices]

    def get_batch_slot_iterator(self) -> Iterator[Any]:
        """
        各バッチスロット用の無限イテレータを返します．
        dataset_list 内の各シーケンスをランダムにシャッフルし、サンプルを逐次 yield します。
        """
        while True:
            shuffled = self.random_shuffle_list(self.dataset_list)
            for ds in shuffled:
                # ds はイテレータとして扱える（例：SequenceForIter のインスタンス）
                for sample in ds:
                    yield self.augmentation_pipeline(sample)

    def __iter__(self) -> Iterator[Tuple[Tuple[Any, ...], int]]:
        worker_info = torch.utils.data.get_worker_info()
        local_worker_id = 0 if worker_info is None else worker_info.id
        if self.print_seed_debug:
            seed = worker_info.seed if worker_info is not None else None
            global_rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
            local_num_workers = 1 if worker_info is None else worker_info.num_workers
            global_worker_id = global_rank * local_num_workers + local_worker_id
            rnd_number = torch.randn(1)
            print(f'seed: {seed}, global_worker_id: {global_worker_id}, '
                  f'local_worker_id: {local_worker_id}, rnd_number: {rnd_number}', flush=True)
        # バッチサイズ分のイテレータを作成
        iterators = [self.get_batch_slot_iterator() for _ in range(self.batch_size)]
        # 各イテレータから1サンプルずつ取り出しタプルにまとめ、ワーカー ID も付与
        for batch in zip(*iterators):
            yield (batch, local_worker_id)
