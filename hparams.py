import argparse
params={"train_size":10,
        "val_size":10,
        "test_size":10,
        "batch_size":256,
        "nof_epoch":100,
        "lr":0.0001,
        "gpu":False,
        "nof_points":5,
        "embedding_size":128,
        "hiddens":512,
        "nof_lstms":2,
        "dropout":0.1,
        "bidir":False}

params=argparse.Namespace(**params)
