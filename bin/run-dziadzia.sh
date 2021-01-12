#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

python -u DeepSpeech.py --noshow_progressbar \
  --train_files /dziadzia/train.csv \
  --dev_files /dziadzia/dev.csv \
  --test_files /dziadzia/test.csv \
  --train_batch_size 80 \
  --dev_batch_size 80 \
  --test_batch_size 40 \
  --n_hidden 375 \
  --epochs 33 \
  --early_stop False \
  --dropout_rate 0.22 \
  --learning_rate 0.00095 \
  --report_count 100 \
  --audio_sample_rate 16000 \
  --export_dir /dziadzia/results/model_export/ \
  --checkpoint_dir /dziadzia/results/checkout/ \
  --alphabet_config_path /dziadzia/alphabet.txt \
  "$@"

# python -u DeepSpeech.py --noshow_progressbar \
#   --train_files data/ldc93s1/ldc93s1.csv \
#   --test_files data/ldc93s1/ldc93s1.csv \
#   --train_batch_size 1 \
#   --test_batch_size 1 \
#   --n_hidden 100 \
#   --epochs 200 \
#   --checkpoint_dir "$checkpoint_dir" \
#   "$@"
