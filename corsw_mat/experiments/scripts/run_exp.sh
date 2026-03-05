#!/bin/bash

python /root/corsw_mat/experiments/scripts/da_transfs_changeall_Rd_matrix_var.py  --epho 500 --ntry 5 --lr_cov 2.5 --lr_cor 2.5 --lr_mix 0.5 --power 0.25 --distance olm --task subject --use_cov_net 1 --use_cor_net 1

python /root/corsw_mat/experiments/scripts/da_transfs_changeall_Rd_matrix_var.py  --epho 500 --ntry 5 --lr_cov 2.5 --lr_cor 2.5 --lr_mix 0.5 --power 0.25 --distance ecm --task subject --use_cov_net 1 --use_cor_net 1

python /root/corsw_mat/experiments/scripts/da_transfs_changeall_Rd_matrix_var.py  --epho 500 --ntry 5 --lr_cov 2.5 --lr_cor 2.5 --lr_mix 0.5 --power 0.25 --distance lecm --task subject --use_cov_net 1 --use_cor_net 1
