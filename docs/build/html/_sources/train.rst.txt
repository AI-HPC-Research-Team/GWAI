###################################################
Training examples of AI-centered model
###################################################

.. autosummary::
   :toctree: generated


===========================================
Signal Classification
===========================================

Firstly, activating ``waveform`` environment.
Then, by running `train_classify.py <https://github.com/AI-HPC-Research-Team/GWAI/tree/main/demos/train_classify.py>`_ script, your own signal classification model can be trained. 

.. code-block:: console
    :linenos:

    $ conda activate waveform
    $ cd /workspace/GWAI/demos
    $ python train_classify.py

You can modify `classify.yaml <https://github.com/AI-HPC-Research-Team/GWAI/tree/main/configs/classify.yaml>`_ to define your own training dataset as well as model configurations. For example:

.. code-block:: yaml
    :linenos:

    dataset:
    save_path: "../datasets/classify/"
    fn: emri_asd_test.hdf5
    dataloader:
    batch_size: 256
    num_workers: 8

    training:
    test_only: False
    checkpoint_dir: 
    gpu: 0
    n_epoch: 50
    # loss_fn: "bce_with_logits"
    loss_fn: "cross_entropy"
    optimizer_type: "adam"
    optimizer_kwargs:
        lr: 5e-5
        weight_decay: 1e-3
    scheduler_type: "plateau"
    scheduler_kwargs:
        mode: "min"
        factor: 0.5
        patience: 5
        threshold: 1e-4
    result_dir: "./results//${now:%Y-%m-%d}/${now:%H-%M-%S}"
    result_fn: "inf_result.npy"
    use_wandb: False

    net:
    input_channels: 2
    n_classes: 2
    n_hidden: 128
    n_levels: 10
    kernel_size: 3
    num_classes: 2
    dropout: 0

The output log can be seen as follows.

.. code-block:: shell
    :linenos:

      [2024-02-04 10:25:46,915][nn.dataloader][INFO] - Loading data from ../datasets/detection/emri_asd_test.hdf5
      Using Adam optimizer, lr=5e-05, weight_decay=0.001
      Total parameters: 940.42K
      Trainable parameters: 940.42K
      Non-trainable parameters: 0
      Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:01<00:00, 138.53it/s, loss=6.94e-01, acc=0.49]                                                                                                                                                                                                 | 0/200 [00:00<?, ?it/s]Time: 0.010484933853149414
      100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 223.66it/s, loss=6.91e-01, acc=0.5050]
      [2024-02-04 10:25:54,895][nn.trainer][INFO] - EPOCH 1   : lr=5.00e-05,   train_loss=6.94e-01,    train_acc=0.4900,       val_loss=6.91e-01       valid_acc=0.5050
      Epoch 2: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:01<00:00, 156.30it/s, loss=6.91e-01, acc=0.50]
      0%|                                                                                                                                                                                                  | 0/200 [00:00<?, ?it/s]Time: 0.010904073715209961

==============================================
Data Denoising
==============================================

Firstly, downloading demo dataset (``train_data, valid_data, test_data``) from `this repository <https://github.com/AI-HPC-Research-Team/LIGO_noise_suppression>`_.
and put it under `datasets/denoise <https://github.com/AI-HPC-Research-Team/GWAI/tree/main/datasets/denoise>`_ folder.
By running `denoise_demo.sh <https://github.com/AI-HPC-Research-Team/GWAI/tree/main/demo/denoise_demo.sh>`_ script, your own denoising model can be trained. 

You can modify configurations in `denoise_demo.sh <https://github.com/AI-HPC-Research-Team/GWAI/tree/main/demo/denoise_demo.sh>`_ to build your own model with different model size.

.. code-block:: console
    :linenos:

    $ conda activate base
    $ cd /workspace/GWAI/demos
    $ bash denoise_demo.sh

The training parameters can be modified in `denoise_demo.sh`, for example:

.. code-block:: shell
    :linenos:

    #!/bin/bash

    GPUS_PER_NODE=2
    MASTER_ADDR=localhost
    MASTER_PORT=6066
    NNODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
    DATA_PATH=../dataset/denoise

    DETS=H1
    CHECKPOINT_PATH=demo

    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

    export CUDA_VISIBLE_DEVICES=6,7
    python -m torch.distributed.launch $DISTRIBUTED_ARGS \
        pretrain_gw.py \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --num-layers 16 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --micro-batch-size 8 \
        --segment-length 256 \
        --dets $DETS \
        --seq-length 128 \
        --max-position-embeddings 128 \
        --train-iters 30000 \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 0.0001 \
        --lr-decay-style linear \
        --min-lr 1.0e-5 \
        --lr-decay-iters 9900 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --lr-warmup-fraction .002 \
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 1 \
        --dataloader-type cyclic \
        --fp16 \
        --no-binary-head

The output log can be seen as follows.

.. code-block:: shell
    :linenos:

      using world size: 2, data-parallel-size: 2, tensor-model-parallel size: 1, pipeline-model-parallel size: 1 
      setting global batch size to 16
      using torch.float16 for parameters ...
      ------------------------ arguments ------------------------
      accumulate_allreduce_grads_in_fp32 .............. False
      adam_beta1 ...................................... 0.9
      xxxxxxx
      -------------------- end of arguments ---------------------
      setting number of micro-batches to constant 1
      > initializing torch distributed ...
      > initializing tensor model parallel with size 1
      > initializing pipeline model parallel with size 1
      > setting random seeds to 1234 ...
      > initializing model parallel cuda seeds on global rank 0, model parallel rank 0, and data parallel rank 0 with model parallel seed: 3952 and data parallel seed: 1234
      > compiling and loading fused kernels ...
      Detected CUDA files, patching ldflags
      Emitting ninja build file /workspace/GWAI/demo/../src/model/denoising/fused_kernels/build/build.ninja...
      Building extension module scaled_upper_triang_masked_softmax_cuda...
      Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
      ninja: no work to do.
      Loading extension module scaled_upper_triang_masked_softmax_cuda...
      Detected CUDA files, patching ldflags
      Emitting ninja build file /workspace/GWAI/demo/../src/model/denoising/fused_kernels/build/build.ninja...
      Building extension module scaled_masked_softmax_cuda...
      Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
      ninja: no work to do.
      Loading extension module scaled_masked_softmax_cuda...
      Detected CUDA files, patching ldflags
      Emitting ninja build file /workspace/GWAI/demo/../src/model/denoising/fused_kernels/build/build.ninja...
      Building extension module fused_mix_prec_layer_norm_cuda...
      Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
      ninja: no work to do.
      Loading extension module fused_mix_prec_layer_norm_cuda...
      >>> done with compiling and loading fused kernels. Compilation time: 3.274 seconds
      time to initialize megatron (seconds): 41.829
      [after megatron is initialized] datetime: 2024-02-02 15:50:01 
      building WaveFormer model ...
      > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 220058673
      > learning rate decay style: linear
      WARNING: could not find the metadata file demo/latest_checkpointed_iteration.txt 
         will not load any checkpoints and will start from random
      time (ms) | load-checkpoint: 0.16
      [after model, optimizer, and learning rate scheduler are built] datetime: 2024-02-02 15:50:01 
      > building train, validation, and test datasets ...
      > building train, validation, and test datasets for BERT ...
      > finished creating BERT datasets ...
      [after dataloaders are built] datetime: 2024-02-02 15:50:06 
      done with setup ...time (ms) | model-and-optimizer-setup: 111.39 | train/valid/test-data-iterators-setup: 4415.50

      training ...
      [before the start of training step] datetime: 2024-02-02 15:50:06 
      iteration        1/   30000 | current time: 1706860208.35 | consumed samples:           16 | elapsed time per iteration (ms): 1996.1 | learning rate: 0.000E+00 | global batch size:    16 | loss scale: 4294967296.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
      time (ms) | backward-compute: 138.46 | backward-params-all-reduce: 32.71 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 3.17 | optimizer-unscale-and-check-inf: 42.67 | optimizer: 45.94 | batch-generator: 263.80
      ----------------------------------------------------------------------------------------------------
      validation loss at iteration 1 | lm loss value: 4.280033E-01 | lm loss PPL: 1.534191E+00 | 
      --------------------------------------------------------------------------------------------
      iteration        2/   30000 | current time: 1706860208.78 | consumed samples:           32 | elapsed time per iteration (ms): 429.4 | learning rate: 0.000E+00 | global batch size:    16 | loss scale: 2147483648.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
      time (ms) | backward-compute: 31.50 | backward-params-all-reduce: 35.43 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 2.87 | optimizer-unscale-and-check-inf: 12.14 | optimizer: 15.32 | batch-generator: 274.37
      ----------------------------------------------------------------------------------------------------
      validation loss at iteration 2 | lm loss value: 4.258614E-01 | lm loss PPL: 1.530909E+00 | 
      --------------------------------------------------------------------------------------------


==============================================
Signal Detection
==============================================

Firstly, activating ``waveform`` environment.
Then, by running `train_detection.py <https://github.com/AI-HPC-Research-Team/GWAI/tree/main/demos/train_detection.py>`_ script, your own detection model can be trained. 

.. code-block:: console
    :linenos:

    $ conda activate waveform
    $ cd /workspace/GWAI/
    $ python demos/train_detection.py configs/detection.yaml

You can modify `detection.yaml` to define your own training dataset as well as model configurations. For example:

.. code-block:: yaml
    :linenos:

    # Basic parameters
    # Seed needs to be set at top of yaml, before objects with parameters are made
    #
    seed: 1607
    __set_seed: !apply:torch.manual_seed [!ref <seed>]

    # cuda device num
    cuda: 5
    # Data params
    data_folder: './datasets/detection'
    data_hdf5: smbhb_test.hdf5
    noise_hdf5: noise_test.hdf5

    experiment_name: detection_demo
    #----------------------------------------

    output_folder: !ref results/<experiment_name>/<seed>
    train_log: !ref <output_folder>/train_log.txt
    save_folder: !ref <output_folder>/save

    # Experiment params
    auto_mix_prec: False
    test_only: False
    num_spks: 1
    progressbar: True
    save_inf_data: False
    save_attention_weights: False
    # se loss * alpha + clsf loss * (1 - alpha)
    alpha: 1
    inf_data: !ref <save_folder>/inf_test/
    # att_data: !ref <save_folder>/inf_test/

    # Training parameters
    N_epochs: 100
    batch_size: 16
    lr: 0.0005
    clip_grad_norm: 5
    loss_upper_lim: 999999  # this is the upper limit for an acceptable loss
    # if True, the training sequences are cut to a specified length
    limit_training_signal_len: False
    # this is the length of sequences if we choose to limit
    # the signal length of training sequences
    training_signal_len: 4000
    dataloader_opts:
        batch_size: !ref <batch_size>
        num_workers: 3

    # loss thresholding -- this thresholds the training loss
    threshold_byloss: True
    threshold: -50

    # Encoder parameters
    N_encoder_out: 256
    out_channels: 256
    kernel_size: 16
    kernel_stride: 8


    # Specifying the network
    Encoder: !new:speechbrain.lobes.models.dual_path.Encoder
        kernel_size: !ref <kernel_size>
        out_channels: !ref <N_encoder_out>


    SBtfintra: !new:speechbrain.lobes.models.dual_path.SBTransformerBlock
        num_layers: 2
        d_model: !ref <out_channels>
        nhead: 4
        d_ffn: 256
        dropout: 0
        use_positional_encoding: True
        norm_before: True

    SBtfinter: !new:speechbrain.lobes.models.dual_path.SBTransformerBlock
        num_layers: 2
        d_model: !ref <out_channels>
        nhead: 4
        d_ffn: 256
        dropout: 0
        use_positional_encoding: True
        norm_before: True

    MaskNet: !new:speechbrain.lobes.models.dual_path.Dual_Path_Model
        num_spks: !ref <num_spks>
        in_channels: !ref <N_encoder_out>
        out_channels: !ref <out_channels>
        num_layers: 2
        K: 25
        intra_model: !ref <SBtfintra>
        inter_model: !ref <SBtfinter>
        norm: ln
        linear_layer_after_inter_intra: False
        skip_around_intra: True

    Decoder: !new:speechbrain.lobes.models.dual_path.Decoder
        in_channels: !ref <N_encoder_out>
        out_channels: 1
        kernel_size: !ref <kernel_size>
        stride: !ref <kernel_stride>
        bias: False

    linear_1: !new:speechbrain.nnet.linear.Linear
        input_size: !ref <training_signal_len>
        n_neurons: 512

    relu: !new:torch.nn.ReLU

    linear_2: !new:speechbrain.nnet.linear.Linear
        input_size: 512
        n_neurons: 1

    optimizer: !name:torch.optim.Adam
        lr: !ref <lr>
        weight_decay: 0


    loss: !name:speechbrain.nnet.losses.get_si_snr_with_pitwrapper
    loss2: !name:speechbrain.nnet.losses.bce_loss

    lr_scheduler: !new:speechbrain.nnet.schedulers.ReduceLROnPlateau
        factor: 0.5
        patience: 2
        dont_halve_until_epoch: 35

    epoch_counter: !new:speechbrain.utils.epoch_loop.EpochCounter
        limit: !ref <N_epochs>

    modules:
        encoder: !ref <Encoder>
        decoder: !ref <Decoder>
        masknet: !ref <MaskNet>
        linear_1: !ref <linear_1>
        linear_2: !ref <linear_2>

    checkpointer: !new:speechbrain.utils.checkpoints.Checkpointer
        checkpoints_dir: !ref <save_folder>
        recoverables:
            encoder: !ref <Encoder>
            decoder: !ref <Decoder>
            masknet: !ref <MaskNet>
            linear_1: !ref <linear_1>
            linear_2: !ref <linear_2>
            counter: !ref <epoch_counter>
            lr_scheduler: !ref <lr_scheduler>
            # mlp: !ref <MLP>

    train_logger: !new:speechbrain.utils.train_logger.FileTrainLogger
        save_file: !ref <train_log>


The output log can be seen as follows.

.. code-block:: shell
    :linenos:

      speechbrain.core - Beginning experiment!
      speechbrain.core - Experiment folder: results/detection_demo22/1607
      speechbrain.core - Info: test_only arg overridden by command line input to: False
      speechbrain.core - Info: auto_mix_prec arg from hparam file is used
      speechbrain.core - 5.6M trainable parameters in Separation
      speechbrain.utils.checkpoints - Would load a checkpoint here, but none found yet.
      speechbrain.utils.epoch_loop - Going into epoch 1
      100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:02<00:00,  5.45it/s, loss1=6.18, loss2=0.693, train_loss=6.18]
      100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.50it/s]
      speechbrain.utils.train_logger - epoch: 1, lr: 5.00e-04 - train si-snr: 6.18, train loss1: 6.18, train loss2: 6.93e-01 - valid si-snr: -6.32e-01, valid loss1: -6.32e-01, valid loss2: 6.96e-01
      speechbrain.utils.checkpoints - Saved an end-of-epoch checkpoint in results/detection_demo22/1607/save/CKPT+2024-02-02+15-55-58+00
      speechbrain.utils.epoch_loop - Going into epoch 2
      100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:02<00:00,  5.72it/s, loss1=-2.26, loss2=0.693, train_loss=-2.26]
      100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.47it/s]
      speechbrain.utils.train_logger - epoch: 2, lr: 5.00e-04 - train si-snr: -2.26e+00, train loss1: -2.26e+00, train loss2: 6.93e-01 - valid si-snr: -2.13e+00, valid loss1: -2.13e+00, valid loss2: 6.97e-01
      speechbrain.utils.checkpoints - Saved an end-of-epoch checkpoint in results/detection_demo22/1607/save/CKPT+2024-02-02+15-56-01+00
      speechbrain.utils.checkpoints - Deleted checkpoint in results/detection_demo22/1607/save/CKPT+2024-02-02+15-55-58+00
      speechbrain.utils.epoch_loop - Going into epoch 3


