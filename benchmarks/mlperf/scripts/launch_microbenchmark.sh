echo "config: ${config}"
source ./${config}.sh
export run_name=${model_name}_${tpu}_${attention}_ici_${ici_tensor_parallelism}-${ici_autoregressive_parallelism}_${reshape_q}_${quant_mode}_pbs${per_device_batch_size}_${compute_axis_order//,/}-${prefill_cache_axis_order//,/}-${ar_cache_axis_order//,/}

export inference_microbenchmark_stages=${inference_microbenchmark_stages:="prefill,generate"}
export inference_microbenchmark_prefill_lengths=${inference_microbenchmark_prefill_lengths:="64,128,256,512,1024"}
echo "inference_microbenchmark_stages: ${inference_microbenchmark_stages}"
echo "inference_microbenchmark_prefill_lengths: ${inference_microbenchmark_prefill_lengths}"

cd /maxtext
export run_dir=${base_output_dir}/microbenchmark/${run_name}/${experiment_time}/
echo "run_dir: ${run_dir}"
gsutil cp ${config_file_path} ${run_dir}/

python3 MaxText/inference_microbenchmark.py \
    ${config_file_path} \
    model_name=${model_name} \
    tokenizer_path=assets/tokenizer.llama2 \
    load_parameters_path=${checkpoint_path} \
    async_checkpointing=false \
    weight_dtype=bfloat16 \
    attention=dot_product \
    reshape_q=${reshape_q} \
    scan_layers=false \
    max_prefill_predict_length=1024 \
    max_target_length=2048 \
    base_output_directory=${base_output_dir}/microbenchmark \
    run_name=${run_name}/${experiment_time} \
    save_config_to_gcs=true \
    profiler=xplane \
    enable_single_controller=true \
    ici_tensor_parallelism=${ici_tensor_parallelism} \
    ici_autoregressive_parallelism=${ici_autoregressive_parallelism} \
    allow_split_physical_axes=${allow_split_physical_axes} \
    inference_microbenchmark_prefill_lengths=${inference_microbenchmark_prefill_lengths} \
    inference_microbenchmark_stages=${inference_microbenchmark_stages} \
    inference_microbenchmark_loop_iters=10 \
    per_device_batch_size=${per_device_batch_size} \
    quantization=${quantization} \
    quantize_kvcache=${quantize_kvcache} \
    kv_quant_axis=${kv_quant_axis} \
    kv_quant_dtype=${kv_quant_dtype} \
    checkpoint_is_quantized=${checkpoint_is_quantized} \
    compute_axis_order=${compute_axis_order} \
    prefill_cache_axis_order=${prefill_cache_axis_order} \
    ar_cache_axis_order=${ar_cache_axis_order} 2>&1 | tee results.log && gsutil mv results.log ${run_dir}/
