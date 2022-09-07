test_file=$1
pred=$2
python preprocess.py --train $test_file --valid $test_file

python run_summarization.py --model_name_or_path ./model \
--output_dir generate \
--per_gpu_train_batch_size 1 \
--text_column maintext \
--summary_column title \
--gradient_accumulation_steps 2 \
--learning_rate 4e-5 \
--predict_with_generate \
--source_prefix "summarize: " \
--overwrite_output_dir \
--train_file train.json \
--do_predict \
--test_file valid.json \
--out_file $pred \
--num_beams 5
