#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=30
stage=0
stop_stage=100

num_splits=10

# We assume dl_dir (download dir) contains the following
# directories and files. If not, you need to apply aishell2 through
# their official website.
# https://www.aishelltech.com/aishell_2
#
#  - $dl_dir/aishell2
#
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech

dl_dir=$PWD/download
lang_char_dir=data/lang_char

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "stage 0: Download data"

  # If you have pre-downloaded it to /path/to/aishell2,
  # you can create a symlink
  #
  #   ln -sfv /path/to/aishell2 $dl_dir/aishell2
  #
  # The directory structure is
  # aishell2/
  # |-- AISHELL-2
  # |   |-- iOS
  #         |-- data
  #             |-- wav
  #             |-- trans.txt
  #         |-- dev
  #             |-- wav
  #             |-- trans.txt
  #         |-- test
  #             |-- wav
  #             |-- trans.txt


  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  #
  #   ln -sfv /path/to/musan $dl_dir/musan
  #
  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare aishell2 manifest"
  # We assume that you have downloaded and unzip the aishell2 corpus
  # to $dl_dir/aishell2
  if [ ! -f data/manifests/.aishell2_manifests.done ]; then
    mkdir -p data/manifests
    lhotse prepare aishell2 $dl_dir/aishell2 data/manifests -j $nj
    touch data/manifests/.aishell2_manifests.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  if [ ! -f data/manifests/.musan_manifests.done ]; then
    log "It may take 6 minutes"
    mkdir -p data/manifests
    lhotse prepare musan $dl_dir/musan data/manifests
    touch data/manifests/.musan_manifests.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Preprocess aishell2 manifest"
  if [ ! -f data/fbank/.preprocess_complete ]; then
    python3 ./local/preprocess_aishell2.py
    touch data/fbank/.preprocess_complete
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute features for DEV and TEST subsets of aishell2"
  python3 ./local/compute_fbank_aishell2_dev_test.py
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Split train subset into ${num_splits} pieces"
  split_dir=data/fbank/train_split_${num_splits}
  if [ ! -f $split_dir/.split_completed ]; then
    lhotse split $num_splits ./data/fbank/aishell2_cuts_train_raw.jsonl.gz $split_dir
    touch $split_dir/.split_completed
  fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Compute features for train"
  ./local/compute_fbank_aishell2_splits.py \
    --training-subset train \
    --num-workers 2 \
    --batch-duration 1000 \
    --start 0 \
    --num-splits $num_splits
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 11: Combine features for train"
  if [ ! -f data/fbank/cuts_train.jsonl.gz ]; then
    pieces=$(find data/fbank/train_split_${num_splits} -name "cuts_train.*.jsonl.gz")
    lhotse combine $pieces data/fbank/aishell2_cuts_train.jsonl.gz
  fi
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 14: Compute fbank for musan"
  mkdir -p data/fbank
  ./local/compute_fbank_musan.py
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Prepare char based lang"
  mkdir -p $lang_char_dir

  if ! which jq; then
      echo "This script is intended to be used with jq but you have not installed jq
      Note: in Linux, you can install jq with the following command:
      1. wget -O jq https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64
      2. chmod +x ./jq
      3. cp jq /usr/bin" && exit 1
  fi
  if [ ! -f $lang_char_dir/text ] || [ ! -s $lang_char_dir/text ]; then
    log "Prepare text."
    gunzip -c data/manifests/aishell2_supervisions_train.jsonl.gz \
      | jq '.text' | sed 's/"//g' \
      | ./local/text2token.py -t "char" > $lang_char_dir/text
  fi

  # The implementation of chinese word segmentation for text,
  # and it will take about 15 minutes.
  if [ ! -f $lang_char_dir/text_words_segmentation ]; then
    python3 ./local/text2segments.py \
      --num-process $nj \
      --input-file $lang_char_dir/text \
      --output-file $lang_char_dir/text_words_segmentation
  fi

  cat $lang_char_dir/text_words_segmentation | sed 's/ /\n/g' \
    | sort -u | sed '/^$/d' | uniq > $lang_char_dir/words_no_ids.txt

  if [ ! -f $lang_char_dir/words.txt ]; then
    python3 ./local/prepare_words.py \
      --input-file $lang_char_dir/words_no_ids.txt \
      --output-file $lang_char_dir/words.txt
  fi
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
  log "Stage 10: Prepare char based L_disambig.pt"
  if [ ! -f data/lang_char/L_disambig.pt ]; then
    python3 ./local/prepare_char.py \
      --lang-dir data/lang_char
  fi
fi

