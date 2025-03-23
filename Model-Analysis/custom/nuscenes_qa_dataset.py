
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS

@DATASETS.register_module()
class NuScenesQADataset(NuScenesDataset):
    def __init__(self, qa_sample_tokens=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if qa_sample_tokens:
            with open(qa_sample_tokens) as f:
                token_set = set(line.strip() for line in f)
            original_len = len(self.data_infos)
            self.data_infos = [info for info in self.data_infos if info['token'] in token_set]
            print(f"Filtered {original_len} → {len(self.data_infos)} QA samples")
