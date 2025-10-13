docker pull docker.io/khashayargatmiry/open-instruct-dev:latest

docker run --rm -it \                                                    
  --platform=linux/amd64 \
  -e HUGGINGFACE_HUB_TOKEN="XXX" \
  -v "$PWD":/open-instruct-repo -w /open-instruct-repo \
  khashayargatmiry/open-instruct-dev:latest \
  bash -lc '
    huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN" --add-to-git-credential && \
    accelerate launch \
      --mixed_precision no \
      --num_processes 1 \
      --num_machines 1 \
      open_instruct/finetune.py \
        --exp_name olmo2_7b_sft \
        --model_name_or_path allenai/OLMo-2-1124-7B \
        --tokenizer_name allenai/OLMo-2-1124-7B \
        --add_bos \
        --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 1.0 \
        --max_seq_length 2048 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 \
        --num_train_epochs 1 \
        --report_to none --with_tracking False \
        --output_dir ./runs/olmo2-sft-local'