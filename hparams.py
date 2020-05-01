import argparse

"""
Specify Model Parameters here
"""
model_params={
        "lr": 1,
        "gpu": True,
        "embedding_size": 128,
        "hiddens": 256,
        "nof_lstms": 2,
        "dropout": 0.1,
        "bidir": False,
        "name":"model_5.pt"
}

model_params = argparse.Namespace(**model_params)
"""
Specify Training Parameters here
"""
train_params = {
    "train_size": 1000000,
    "val_size": 1000,
    "batch_size": 256,
    "nof_epoch": 2,
    "nof_points": 5,
    "file":"./data/tsp5.txt",
    "hyperdash":False,
}

train_params = argparse.Namespace(**train_params)


"""
Specify Testing Parameters here
"""
test_params={
        "test_size":10,
        "batch_size":16,
        "nof_points": 5,
        "model":"./model/model_5.pt",
        "file":"./data/tsp5_test.txt",
}

test_params = argparse.Namespace(**test_params)
