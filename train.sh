torchrun --nproc_per_node=$NUMBER_OF_GPUS --nnodes=1 -m train_mlm --device_bsz $BATCH_SIZE
