#  bin#!/bin/bash
# python runfile.py -Data FullDataset -GPU 1 -DFF 1024 -NEL 4 -NDL 4 -ES 256 -W False
# increase embedding
python runfile.py -Data FullDataset-larger -GPU 1 -DFF 256 -NEL 10 -NDL 10 -ES 32 -W True -SI 0,1,4,6,8,12
# python runlinearmlp.py
