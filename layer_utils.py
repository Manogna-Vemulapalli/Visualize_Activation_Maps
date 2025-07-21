from keras.models import Model

def get_intermediate_layers(model):
    """
    Returns a dictionary of intermediate layer models to extract feature maps.
    Only Conv2D and MaxPooling2D layers are included.
    """
    layer_outputs = {}
    for layer in model.layers:
        if 'conv' in layer.name or 'pool' in layer.name:
            intermediate_model = Model(inputs=model.input, outputs=layer.output)
            layer_outputs[layer.name] = intermediate_model
    return layer_outputs
