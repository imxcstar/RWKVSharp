import numpy as np
import torch
import opslist
import convert
class RWKVConverter:
    def __init__(self, src_path: str, dest_path: str,name: str, data_type: str):
        self.src_path = src_path
        self.dest_path = dest_path
        self.data_type = data_type if data_type in ['FP16', 'FP32', 'float16', 'float32'] else 'FP16'
        self.name=name
    @staticmethod
    def convert_model(path, dtype,savePath,name):
        w = torch.load(path, map_location="cpu")
        dims = len(w["blocks.0.att.key.weight"])
        layers = len(list(filter(lambda x: "blocks" in x and "ln1.bias" in x, w.keys())))
        ops = opslist.RWKVOnnxOps(layers,dims,dtype=dtype, opsVersion=17 if 'world' in name.lower() else 15, useSafeWKV=True, externalData=True, splitExternalData=False, fp32inout=True,savePath=savePath,cname=name)
        convert.RnnRWKV(ops,w)
    def convert(self):
        print(f'Reading {self.src_path}')
        self.convert_model(self.src_path, np.float16 if self.data_type in ['FP16', 'float16'] else np.float32 ,self.dest_path,self.name)
        print('Done')