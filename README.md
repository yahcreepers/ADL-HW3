# Homework 3 ADL NTU 2022

## Environment

```shell
./download.sh
```

## Preprocessing

```shell
python preprocess.py --train <train_file> --valid <test_file>
```

## Summarization

重現Training過程

```shell
python run_summarization.py --model_name_or_path google/mt5-small --output_dir out_T5 --per_gpu_train_batch_size 1  --text_column maintext --summary_column title --save_steps 20000 --gradient_accumulation_steps 2 --evaluation_strategy steps --eval_steps 20000 --learning_rate 4e-5 --source_prefix "summarize: " --predict_with_generate --num_train_epochs 30 --do_train --do_eval --overwrite_output_dir --train_file train.json --validation_file valid.json
```

## Results

| Model            | Rouge-1            | Rouge-2             | Rouge-L             |
| ---------------- | ------------------ | ------------------- | ------------------- |
| google/mt5-small | 0.2557504722089096 | 0.10335435105592458 | 0.22969651422290516 |
