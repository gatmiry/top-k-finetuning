docker pull docker.io/khashayargatmiry/open-instruct-dev:latest

export HUGGINGFACE_HUB_TOKEN=XXX

docker run --rm --gpus all \
  -e HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN}" \
  -v "$PWD":/workspace \
  khashayargatmiry/open-instruct-dev:latest \
  bash -lc '
    cd /workspace && \
    huggingface-cli login --token "${HUGGINGFACE_HUB_TOKEN}" --add-to-git-credential && \
    accelerate launch \
      --mixed_precision bf16 \
      --num_processes 1 \
      --num_machines 1 \
      open_instruct/finetune.py \
        --exp_name olmo2_7b_sft \
        --model_name_or_path allenai/OLMo-2-1124-7B \
        --tokenizer_name allenai/OLMo-2-1124-7B \
        --add_bos \
        --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 1.0 \
        --max_train_samples 100 \
        --dataset_skip_cache \
        --max_seq_length 512 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --learning_rate 2e-5 \
        --num_train_epochs 1 \
        --report_to none --with_tracking False \
        --output_dir ./runs/olmo2-sft-gpu
  '
