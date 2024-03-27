import os
import pickle
from pathlib import Path

import numpy


def load():
    dataset_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    paths = ["batches.meta", *[f"data_batch_{i + 1}" for i in range(5)], "test_batch"]
    paths = [dataset_dir / Path(path) for path in paths]

    files = {}
    for path in paths:
        with open(path, "rb") as fo:
            files[str(path.name)] = pickle.load(fo, encoding="bytes")

    classes = [
        str(name, encoding="utf-8") for name in files["batches.meta"][b"label_names"]
    ]

    x_train = numpy.array(
        [
            files[file][b"data"].reshape(-1, 3, 32, 32) / 255.0
            for file in (f"data_batch_{i + 1}" for i in range(5))
        ]
    ).reshape(-1, 3, 32, 32)
    t_train = numpy.array(
        [files[file][b"labels"] for file in (f"data_batch_{i + 1}" for i in range(5))]
    ).reshape(-1)

    x_test = files["test_batch"][b"data"].reshape(-1, 3, 32, 32) / 255.0
    t_test = numpy.array(files["test_batch"][b"labels"])

    return classes, (x_train, t_train), (x_test, t_test)
