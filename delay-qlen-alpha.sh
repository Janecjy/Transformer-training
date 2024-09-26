python runfile.py -Data FullDataset -GPU 0 -DFF 64 -NEL 5 -NDL 5 -NH 4 -ES 16 -W True -LR 1e-5 -SI 4,8 -LW 200,10 -A 20000 &
python runfile.py -Data FullDataset -GPU 1 -DFF 64 -NEL 5 -NDL 5 -NH 4 -ES 16 -W True -LR 1e-5 -SI 4,8 -LW 200,10 -A 10000 &
python runfile.py -Data FullDataset -GPU 2 -DFF 64 -NEL 5 -NDL 5 -NH 4 -ES 16 -W True -LR 1e-5 -SI 4,8 -LW 200,10 -A 50000 &
wait