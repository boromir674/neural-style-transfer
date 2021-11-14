""" 
"""
def vgg_layers():
    """The network's layer structure of the vgg image model."""
    return (
        (0, 'conv1_1') ,  # (3, 3, 3, 64)
        (1, 'relu1_1') ,
        (2, 'conv1_2') ,  # (3, 3, 64, 64)
        (3, 'relu1_2') ,
        (4, 'pool1')   ,
        (5, 'conv2_1') ,  # (3, 3, 64, 128)
        (6, 'relu2_1') ,
        (7, 'conv2_2') ,  # (3, 3, 128, 128)
        (8, 'relu2_2') ,
        (9, 'pool2')   ,
        (10, 'conv3_1'),  # (3, 3, 128, 256)
        (11, 'relu3_1'),
        (12, 'conv3_2'),  # (3, 3, 256, 256)
        (13, 'relu3_2'),
        (14, 'conv3_3'),  # (3, 3, 256, 256)
        (15, 'relu3_3'),
        (16, 'conv3_4'),  # (3, 3, 256, 256)
        (17, 'relu3_4'),
        (18, 'pool3')  ,
        (19, 'conv4_1'),  # (3, 3, 256, 512)
        (20, 'relu4_1'),
        (21, 'conv4_2'),  # (3, 3, 512, 512)
        (22, 'relu4_2'),
        (23, 'conv4_3'),  # (3, 3, 512, 512)
        (24, 'relu4_3'),
        (25, 'conv4_4'),  # (3, 3, 512, 512)
        (26, 'relu4_4'),
        (27, 'pool4')  ,
        (28, 'conv5_1'),  # (3, 3, 512, 512)
        (29, 'relu5_1'),
        (30, 'conv5_2'),  # (3, 3, 512, 512)
        (31, 'relu5_2'),
        (32, 'conv5_3'),  # (3, 3, 512, 512)
        (33, 'relu5_3'),
        (34, 'conv5_4'),  # (3, 3, 512, 512)
#         35 is relu
#         36 is maxpool
#         37 is fullyconnected (7, 7, 512, 4096)
#         38 is relu
#         39 is fullyconnected (1, 1, 4096, 4096)
#         40 is relu
#         41 is fullyconnected (1, 1, 4096, 1000)
#         42 is softmax
    )


LAYERS = tuple((layer_id for _, layer_id in vgg_layers()))
