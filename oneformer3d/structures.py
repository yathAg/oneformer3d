from collections.abc import Sized
from mmengine.structures import InstanceData


class InstanceData_(InstanceData):
    """We only remove a single assert from __setattr__."""

    def __setattr__(self, name: str, value: Sized):
        """setattr is only used to set data.

        The value must have the attribute of `__len__` and have the same length
        of `InstanceData`.
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super(InstanceData, self).__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')

        else:
            assert isinstance(value,
                              Sized), 'value must contain `__len__` attribute'

            super(InstanceData, self).__setattr__(name, value)


class ChunkedMask:
    """Lazy mask logits computed from query and mask features."""

    def __init__(self, queries, mask_feats):
        self.queries = queries
        self.mask_feats = mask_feats
        self.shape = (queries.shape[0], mask_feats.shape[0])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        q = self.queries[idx]
        if q.dim() == 1:
            q = q.unsqueeze(0)
        return q @ self.mask_feats.T
