python examples/run_expt.py --dataset iwildcam --algorithm IRM --root_dir data --lr 3e-5 --weight_decay 0 --irm_lambda 1 --n_epochs 30 --seed 0  --irm_lambda 1 --log_dir ./logs/t23_front_IRM_lr_3e-5_weight_decay_0_seed_0_lr_3e-5_epochs_30_lambda_1 &&
python examples/run_expt.py --dataset iwildcam --algorithm IRM --root_dir data --lr 3e-5 --weight_decay 0 --irm_lambda 1 --n_epochs 30 --seed 0  --irm_lambda 10 --log_dir ./logs/t23_front_IRM_lr_3e-5_weight_decay_0_seed_0_lr_3e-5_epochs_30_lambda_10 &&
python examples/run_expt.py --dataset iwildcam --algorithm IRM --root_dir data --lr 3e-5 --weight_decay 0 --irm_lambda 1 --n_epochs 30 --seed 0  --irm_lambda 100 --log_dir ./logs/t23_front_IRM_lr_3e-5_weight_decay_0_seed_0_lr_3e-5_epochs_30_lambda_100 

