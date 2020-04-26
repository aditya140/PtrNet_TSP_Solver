import argparse
model_params={
        "lr": 0.0001,
        "gpu": True,
        "nof_points": 5,
        "embedding_size": 128,
        "hiddens": 512,
        "nof_lstms": 2,
        "dropout": 0.1,
        "bidir": False,
}

model_params = argparse.Namespace(**model_params)

train_params = {
    "train_size": 10,
    "val_size": 10,
    "batch_size": 256,
    "nof_epoch": 100,
    "file":"./data/tsp5.txt",
}

train_params = argparse.Namespace(**train_params)

test_params={
        "batch_size":16,
        "model":"./models/model.pt",
        "file":"./data/tsp5_test.txt",
}

test_params = argparse.Namespace(**test_params)
