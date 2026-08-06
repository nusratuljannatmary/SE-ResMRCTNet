# ========== FLOPs + Params Calculation ==========
def get_flops(model, batch_size=1):
    concrete_func = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + list(model.input_shape[1:]), model.inputs[0].dtype))
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph, run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops if flops is not None else 0

# ========== Run Model Build + Report ==========
seq_len = all_epochs_data.shape[1]
input_dim = all_epochs_data.shape[2]
model = build_transformer_model(seq_len, input_dim)

flops = get_flops(model)
params = model.count_params()

print(f"FLOPs: {flops / 1e6:.2f} MFLOPs")
print(f"Parameters: {params / 1e3:.2f} K")
