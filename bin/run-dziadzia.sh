#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -f "data/ldc93s1/ldc93s1.csv" ]; then
    echo "Downloading and preprocessing LDC93S1 example data, saving in ./data/ldc93s1."
    python -u bin/import_ldc93s1.py ./data/ldc93s1
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ldc93s1"))')
fi

# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

python -u DeepSpeech.py --noshow_progressbar \\
  --train_files /home/stryku/DeepSpeech/data/dziadzia/train.csv \\
  --dev_files /home/stryku/DeepSpeech/data/dziadzia/dev.csv \\
  --test_files /home/stryku/DeepSpeech/data/dziadzia/test.csv \\
  --train_batch_size 80 \\
  --dev_batch_size 80 \\
  --test_batch_size 40 \\
  --n_hidden 375 \\
  --epochs 33 \\
  --early_stop False \\
  --es_mean_th 0.1 \\
  --es_std_th 0.1 \\
  --es_steps 15 \\
  --dropout_rate 0.22 \\
  --learning_rate 0.00095 \\
  --report_count 100 \\
  --audio_sample_rate 16000 \\
  --export_dir /home/stryku/DeepSpeech/data/dziadzia/results/model_export/ \\
  --checkpoint_dir /home/stryku/DeepSpeech/data/dziadzia/results/checkout/ \\
  --alphabet_config_path /home/stryku/DeepSpeech/data/dziadzia/alphabet.txt \\
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
