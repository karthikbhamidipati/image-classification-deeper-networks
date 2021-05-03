from torch import device, cuda

run_device = device("cuda" if cuda.is_available() else "cpu")
