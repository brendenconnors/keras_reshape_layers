import gradio as gr
import keras
import numpy as np

# All reshaping layers and their args, descriptions
layers = {
    "Reshape":{
        "args":["target_shape"],
        "descriptions":["""target_shape: Target shape. Tuple of integers, does not include the
        samples dimension (batch size)."""]
    },
    "Flatten":{
        "args":["data_format"],
        "descriptions":["""data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. 
        channels_last corresponds to inputs with shape (batch, ..., channels) while channels_first corresponds to inputs with shape (batch, channels, ...). 
        It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. 
        If you never set it, then it will be "channels_last"."""]
    },
    "RepeatVector":{
        "args":["n"],
        "descriptions":["n: Integer, repetition factor."]
    },
    "Permute":{
        "args":["dims"],
        "descriptions":["""dims: Tuple of integers.
         Permutation pattern does not include the samples dimension. Indexing starts at 1. 
         For instance, (2, 1) permutes the first and second dimensions of the input."""]
    },
    "Cropping1D":{
        "args":["cropping"],
        "descriptions":["""cropping: Int or tuple of int (length 2) 
        How many units should be trimmed off at the beginning and end of the cropping dimension (axis 1). 
        If a single int is provided, the same value will be used for both."""]
    },
    "Cropping2D":{
        "args":["cropping", "data_format"],
        "descriptions":["""cropping: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
    If int: the same symmetric cropping is applied to height and width.
    If tuple of 2 ints: interpreted as two different symmetric cropping values for height and width: (symmetric_height_crop, symmetric_width_crop).
    If tuple of 2 tuples of 2 ints: interpreted as ((top_crop, bottom_crop), (left_crop, right_crop))""",
    """data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. 
    channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with shape 
    (batch_size, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. 
    If you never set it, then it will be "channels_last"."""],
    },
    "Cropping3D":{
        "args":["cropping", "data_format"],
        "descriptions":["""cropping: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
    If int: the same symmetric cropping is applied to depth, height, and width.
    If tuple of 3 ints: interpreted as two different symmetric cropping values for depth, height, and width: (symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop).
    If tuple of 3 tuples of 2 ints: interpreted as ((left_dim1_crop, right_dim1_crop), (left_dim2_crop, right_dim2_crop), (left_dim3_crop, right_dim3_crop))""",
    """data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape
     (batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels) while channels_first corresponds to inputs with shape 
     (batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. 
     If you never set it, then it will be "channels_last"."""]
    },
    "UpSampling1D":{
        "args":["size"],
        "descriptions":["size: Integer. UpSampling factor."]
    },
    "UpSampling2D":{
        "args":["size", "data_format", "interpolation"],
        "descriptions":["size: Int, or tuple of 2 integers. The UpSampling factors for rows and columns.",
        """data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. 
        channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with 
        shape (batch_size, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. 
        If you never set it, then it will be "channels_last".""",
        """interpolation: A string, one of "area", "bicubic", "bilinear", "gaussian", "lanczos3", "lanczos5", "mitchellcubic", "nearest"."""]
    },
    "UpSampling3D":{
        "args":["size","data_format"],
        "descriptions":["size: Int, or tuple of 3 integers. The UpSampling factors for dim1, dim2 and dim3.",
        """data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. 
        channels_last corresponds to inputs with shape (batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels) while 
        channels_first corresponds to inputs with shape (batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3). 
        It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, 
        then it will be "channels_last"."""]
    },
    "ZeroPadding1D":{
        "args":["padding"],
        "descriptions":["""padding: Int, or tuple of int (length 2), or dictionary. - If int: 
        How many zeros to add at the beginning and end of the padding dimension (axis 1). - 
        If tuple of int (length 2): How many zeros to add at the beginning and the end of the padding dimension ((left_pad, right_pad))."""]
    },
    "ZeroPadding2D":{
        "args":["padding", "data_format"],
        "descriptions":["""padding: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
    If int: the same symmetric padding is applied to height and width.
    If tuple of 2 ints: interpreted as two different symmetric padding values for height and width: (symmetric_height_pad, symmetric_width_pad).
    If tuple of 2 tuples of 2 ints: interpreted as ((top_pad, bottom_pad), (left_pad, right_pad))""",
    """data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. 
    channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with shape 
    (batch_size, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. 
    If you never set it, then it will be "channels_last"."""]
    },
    "ZeroPadding3D":{
        "args":["padding", "data_format"],
        "descriptions":["""padding: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
    If int: the same symmetric padding is applied to height and width.
    If tuple of 3 ints: interpreted as two different symmetric padding values for height and width: (symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad).
    If tuple of 3 tuples of 2 ints: interpreted as ((left_dim1_pad, right_dim1_pad), (left_dim2_pad, right_dim2_pad), (left_dim3_pad, right_dim3_pad))""",
    """data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs
     with shape (batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels) while channels_first corresponds to inputs with shape 
     (batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3). It defaults to the image_data_format value found in your Keras config file 
     at ~/.keras/keras.json. If you never set it, then it will be "channels_last"."""]
    }
}
with gr.Blocks() as demo:
    gr.Markdown(f'![Keras](https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_256,w_256,f_auto,q_auto:eco,dpr_1/x3gdrogoamvuvjemehbr)')
    gr.Markdown("# Reshaping Layers")
    gr.Markdown("""This app allows you to play with various Keras Reshaping layers, and is meant to be a
    supplement to the documentation. You are free to change the layer, tensor/array shape, and arguments associated 
    with that layer. Execution will show you the command used as well as your resulting array/tensor.

    Keras documentation can be found [here](https://keras.io/api/layers/reshaping_layers/).<br>
    App built by [Brenden Connors](https://github.com/brendenconnors).<br>
    Built using keras==2.9.0.

    <br><br><br>""")
    
    with gr.Row():
        with gr.Column():
            layers_dropdown = gr.Dropdown(choices=list(layers.keys()), value="Reshape", label="Keras Layer")
            with gr.Box():
                gr.Markdown("**Please enter desired shape.**")
                desired_shape2d = gr.Dataframe(value = [[2,2]],
                    headers = ["Rows", "Columns"], 
                    row_count=(1, 'fixed'), 
                    col_count=(2, "fixed"), 
                    datatype="number",
                    type = "numpy",
                    interactive=True,
                    visible = False
                    )

                desired_shape3d = gr.Dataframe(value = [[2,2,0]],
                    headers = ["Rows", "Columns", "Depth/Channels"], 
                    row_count=(1, 'fixed'), 
                    col_count=(3, "fixed"), 
                    datatype="number",
                    type = "numpy",
                    interactive=True,
                    visible = True
                    )

                desired_shape4d = gr.Dataframe(value = [[2,2,2,0]],
                    headers = ["Rows", "Columns", "Depth", "Channels"], 
                    row_count=(1, 'fixed'), 
                    col_count=(4, "fixed"), 
                    datatype="number",
                    type = "numpy",
                    interactive=True,
                    visible = False
                    )

                button = gr.Button("Generate Tensor")                   
            input_arr = gr.Textbox(label = "Input Tensor",
            interactive = False,
            value = np.array([[1,2],[3,4]]))
            with gr.Box():
                gr.Markdown("**Layer Args**")
                with gr.Row():
                    arg1 = gr.Textbox(label='target_shape')
                    arg2 = gr.Textbox(label='arg2',visible=False)
                    arg3 = gr.Textbox(label='arg3',visible=False)
                with gr.Row():
                    desc1 = gr.Textbox(label= '', value = layers["Reshape"]["descriptions"][0])
                    desc2 = gr.Textbox(label = '', visible=False)
                    desc3 = gr.Textbox(label = '', visible=False)
            result_button = gr.Button("Execute")
        with gr.Column():
            output = gr.Textbox(label = 'Command Used')
            output2 = gr.Textbox(label = 'Result')

    def generate_arr(layer, data1, data2, data3):
        """
        Create Input tensor
        """
        if '1D' in layer:
            data = data1[0]

        elif '2D' in layer:
            data = data2[0]

        elif '3D' in layer:
            data = data3[0]

        elif layer=="RepeatVector":
            data = data1[0]

        else:
            data = data2[0]


        shape = tuple([int(x) for x in data if int(x)!=0])
        elements = [x+1 for x in range(np.prod(shape))]
        return np.array(elements).reshape(shape)


    def add_dim(layer):
        """
        Adjust dimensions component dependent on layer type
        """
        if '1D' in layer:
            return gr.DataFrame.update(visible=True), gr.DataFrame.update(visible=False), gr.DataFrame.update(visible=False)
        elif '2D' in layer:
            return gr.DataFrame.update(visible=False), gr.DataFrame.update(visible=True), gr.DataFrame.update(visible=False)
        elif '3D' in layer:
            return gr.DataFrame.update(visible=False), gr.DataFrame.update(visible=False), gr.DataFrame.update(visible=True)
        elif layer=="RepeatVector":
            return gr.DataFrame.update(visible=True), gr.DataFrame.update(visible=False), gr.DataFrame.update(visible=False)
        return gr.DataFrame.update(visible=False), gr.DataFrame.update(visible=True), gr.DataFrame.update(visible=False)


    def change_args(layer):
        """
        Change layer args dependent on layer name
        """
        n_args = len(layers[layer]["args"])
        args = layers[layer]["args"]
        descriptions = layers[layer]["descriptions"]
        descriptions = descriptions + ['None']*3
        args = args + ['None']*3
        visible_bool = [True if i<=n_args else False for i in range(1,4)]
        return gr.Textbox.update(label=args[0], visible=visible_bool[0]),\
            gr.Textbox.update(label=args[1], visible=visible_bool[1]),\
                gr.Textbox.update(label=args[2], visible=visible_bool[2]),\
                gr.Textbox.update(value = descriptions[0], visible = visible_bool[0]),\
                gr.Textbox.update(value = descriptions[1], visible = visible_bool[1]),\
                gr.Textbox.update(value = descriptions[2], visible = visible_bool[2])
        
    def create_layer(layer_name, arg1, arg2, arg3):
        """
        Create layer given layer name and args
        """
        args = [arg1, arg2, arg3]
        real_args = [x for x in args if x != '']
        arg_str = ','.join(real_args)

        return f"keras.layers.{layer_name}({arg_str})"


    def execute(layer_name, arg1, arg2, arg3, shape1, shape2, shape3):
        """
        Execute keras reshaping layer given input tensor
        """
        args = [arg1, arg2, arg3]
        real_args = [x for x in args if x != '']
        arg_str = ','.join(real_args)
        try:
            layer = eval(f"keras.layers.{layer_name}({arg_str})")
        except Exception as e:
            return f"Error: {e}"

        def arr(data, layer_name):
            if layer_name == "RepeatVector":
                shape = tuple([int(x) for x in data[0] if int(x)!=0])
            else:
                shape = tuple([1] + [int(x) for x in data[0] if int(x)!=0])
            elements = [x+1 for x in range(np.prod(shape))]
            return np.array(elements).reshape(shape)

        if '1D' in layer_name:
            inp = arr(shape1, layer_name)
        elif '2D' in layer_name:
            inp = arr(shape2, layer_name)
        elif '3D' in layer_name:
            inp = arr(shape3, layer_name)
        elif layer_name=="RepeatVector":
            inp = arr(shape1, layer_name)
        else:
            inp = arr(shape2, layer_name)

        try:
            return layer(inp)
        except Exception as e:
            return e

    # Generate tensor
    button.click(generate_arr, [layers_dropdown, desired_shape2d, desired_shape3d, desired_shape4d], input_arr)

    # All changes dependent on layer selected
    layers_dropdown.change(add_dim, layers_dropdown, [desired_shape2d, desired_shape3d, desired_shape4d])
    layers_dropdown.change(change_args, layers_dropdown, [arg1, arg2, arg3, desc1, desc2, desc3])
    layers_dropdown.change(generate_arr, [layers_dropdown, desired_shape2d, desired_shape3d, desired_shape4d], input_arr)
    
    # Show command used and execute it
    result_button.click(create_layer, [layers_dropdown, arg1, arg2, arg3], output)
    result_button.click(execute, [layers_dropdown, arg1, arg2, arg3, desired_shape2d, desired_shape3d, desired_shape4d], output2)

demo.launch()
