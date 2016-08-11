import nnUtils as nn

model = nn.Sequential([
    nn.SpatialConvolution(64,5,5,padding='SAME'),
    nn.SpatialMaxPooling(2,2),
    nn.BatchNormalization(),
    nn.ReLU(),
    nn.Residual([
        nn.SpatialConvolution(64,3,3,padding='SAME'),
        nn.BatchNormalization(),
        nn.ReLU(),
        nn.SpatialConvolution(64,3,3,padding='SAME')
    ]),
    nn.SpatialMaxPooling(2,2),
    nn.BatchNormalization(),
    nn.ReLU(),
    nn.SpatialConvolution(128,3,3,padding='SAME'),
    nn.ReLU(),
    nn.SpatialConvolution(128,3,3,padding='SAME'),
    nn.SpatialMaxPooling(2,2),
    nn.BatchNormalization(),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Affine(128*4*4,256),
    nn.BatchNormalization(),
    nn.Affine(256,10)
])
