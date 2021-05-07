import logging

from torch import device, cuda

run_device = device("cuda" if cuda.is_available() else "cpu")
logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                  datefmt='%Y-%m-%d %H:%M:%S')
