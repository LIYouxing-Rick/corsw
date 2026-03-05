#!/bin/bash

python da_transfs_50.py  --ntry 5 --task subject

python da_transfs_80.py  --ntry 5 --task session
python da_transfs_80.py  --ntry 5 --task subject

python da_transfs_100.py --ntry 5 --task session
python da_transfs_100.py --ntry 5 --task subject

python da_transfs_200.py --ntry 5 --task session
python da_transfs_200.py --ntry 5 --task subject

python da_transfs_300.py --ntry 5 --task session
python da_transfs_300.py --ntry 5 --task subject

python da_transfs_500.py --ntry 5 --task session
python da_transfs_500.py --ntry 5 --task subject
