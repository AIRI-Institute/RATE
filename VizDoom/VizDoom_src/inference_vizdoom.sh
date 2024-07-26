#!/bin/bash

while getopts ":c:m:a:" opt; do
  case $opt in
    c)
      ckpt_folder="$OPTARG"
      ;;
    m)
      mode="$OPTARG"
      ;;
    a)
      arch_mode="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done
 
#ckpt_folder="$1"
ckpt_dir="VizDoom/VizDoom_checkpoints/${ckpt_folder##*/}"
cd "$ckpt_dir"
# # echo $(pwd)
echo "Selected mode: ${mode%/}"

for dir in */; do
    echo Current directory: "${dir%/}"
    cd $OLDPWD
    
    if [[ "$dir" == *_"${mode%/}"_* && "$dir" == *_"${arch_mode%/}"_* ]]; then

        python3 VizDoom/VizDoom_src/inference/inference_vizdoom.py --model_mode "${mode%/}" \
                                                           --ckpt_name "${dir%/}" \
                                                           --ckpt_chooser 0 \
                                                           --ckpt_folder "${ckpt_folder%/}" \
                                                           --arch_mode "${arch_mode%/}"
    fi

    cd "$ckpt_dir"
done


# VizDoom/VizDoom_src/inference_vizdoom.sh -c 'RATE_my' -m 'RATE' -a 'TrXL'