# https://github.com/tensorflow/tensorflow/issues/32809#issuecomment-849439287
import tensorflow as tf
from tensorflow.keras import Model, Sequential

def get_flops(model):
    # Calculate FLOPs (Floating Point Operations) for the model
    if isinstance(model, (Model, Sequential)):

        from tensorflow.python.profiler.model_analyzer import profile
        from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
        # print('TensorFlow:', tf.__version__)

        forward_pass = tf.function(
            model.call,
            input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
        
        opts = (ProfileOptionBuilder(
                    ProfileOptionBuilder.float_operation())
                    .with_empty_output()
                    .build())
        
        # graph_info = profile(forward_pass.get_concrete_function().graph,
        #                         options=ProfileOptionBuilder.float_operation())
        graph_info = profile(forward_pass.get_concrete_function().graph,
                                options=opts)
        
        # The //2 is necessary since `profile` counts multiply and accumulate
        # as two flops, here we report the total number of multiply accumulate ops
        flops = graph_info.total_float_ops // 2

        return flops / 1e9
    else:
        raise ValueError("Input must be a Keras Model or Sequential instance")

def get_weight(model):
    # Calculate the weight of the model in megabytes (MB)
    # weight_mb = sum([tf.reduce_prod(var.shape) * var.dtype.size for var in model.trainable_variables]) / (1024 ** 2)
    weight_mb = sum([tf.reduce_prod(var.shape) * var.dtype.size for var in model.variables]) / (1024 ** 2)
    return weight_mb
    
def get_nb_parameters(model):
    # Count the total number of parameters in the model
    nb_prams = model.count_params()
    return nb_prams / 1e6  # Convert to millions

def compute_model_metrics(model):
    # Compute various metrics for the model
    if isinstance(model, (Model, Sequential)):
        # Compute the number of parameters
        num_params = get_nb_parameters(model)
        # Compute the weight of the model in MB
        weight_mb = get_weight(model)
        # Compute the number of FLOPs
        flops = get_flops(model)
    else:
        raise ValueError("Input must be a Keras Model or Sequential instance")
    
    # Print the computed metrics
    print("\n======= Model metrics =========")
    print(f"Nb of Params: \t {num_params:.2f} Million")
    print(f"Number of FLOPs: {flops:.2f} Billion")
    print(f"Weight: \t {weight_mb:.2f} MB")
    
