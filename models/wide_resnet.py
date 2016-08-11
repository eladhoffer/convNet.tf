import tensorflow as tf
import nnUtils as nn

def block(inputDim, num_feats=16, k=1, N=1):
    with tf.variable_scope('block'):
        modules = []
        dim = inputDim
        for i in range(N):
            curr_layers = [
                BatchNormalization(),
                ReLU(),
                SpatialConvolution(num_feats*k,[3,3],padding='SAME'),
                BatchNormalization(),
                ReLU(),
                SpatialConvolution(num_feats*k,[3,3],padding='SAME')
            ]

            if dim == num_feats*k:
                modules += [Residual(curr_layers)]
            else:
                modules += curr_layers

            dim = num_feats*k

        return Sequential(modules)


k = 3
N = 4

model = Sequential([
    SpatialConvolution(16,[3,3],padding='SAME'),
    block(16,16,k,N),
    SpatialMaxPooling(2,2),
    block(16*k,32,k,N),
    SpatialMaxPooling(2,2),
    block(32*k,64,k,N),
    SpatialAveragePooling(8,8,1,1),
    BatchNormalization(),
    ReLU(),
    Affine(64*k, 10)
])
