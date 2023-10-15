import argparse
import struct
import torch
from typing import Dict
class RWKVConverter:
    def __init__(self, src_path: str, dest_path: str, data_type: str):
        self.src_path = src_path
        self.dest_path = dest_path
        self.data_type = data_type if data_type in ['FP16', 'FP32', 'float16', 'float32'] else 'FP16'
    @staticmethod
    def get_layer_count(state_dict: Dict[str, torch.Tensor]) -> int:
        n_layer: int = 0
        while f'blocks.{n_layer}.ln1.weight' in state_dict:
            n_layer += 1
        assert n_layer > 0
        return n_layer
    def write_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        emb_weight: torch.Tensor = state_dict['emb.weight']
        n_layer: int = self.get_layer_count(state_dict)
        n_vocab: int = emb_weight.shape[0]
        n_embed: int = emb_weight.shape[1]
        with open(self.dest_path, 'wb') as out_file:
            is_FP16: bool = self.data_type in ['FP16', 'float16']
            out_file.write(struct.pack(
                '=iiiiii',
                0x67676d66,
                101,
                n_vocab,
                n_embed,
                n_layer,
                1 if is_FP16 else 0
            ))
            for k in state_dict.keys():
                tensor: torch.Tensor = state_dict[k].float()
                if '.time_' in k:
                    tensor = tensor.squeeze()
                if '.time_decay' in k:
                    tensor = -torch.exp(tensor)
                if is_FP16 and len(tensor.shape) > 1:
                    tensor = tensor.half()
                shape = tensor.shape
                print(f'Writing {k}, shape {shape}, type {tensor.dtype}')
                k_encoded: bytes = k.encode('utf-8')
                out_file.write(struct.pack(
                    '=iii',
                    len(shape),
                    len(k_encoded),
                    1 if tensor.dtype == torch.float16 else 0
                ))
                for dim in reversed(tensor.shape):
                    out_file.write(struct.pack('=i', dim))
                out_file.write(k_encoded)
                tensor.numpy().tofile(out_file)
    def convert(self):
        print(f'Reading {self.src_path}')
        state_dict: Dict[str, torch.Tensor] = torch.load(self.src_path, map_location='cpu')
        self.write_state_dict(state_dict)
        print('Done')