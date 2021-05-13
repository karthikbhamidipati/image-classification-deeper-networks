import logging
from argparse import ArgumentParser
from logging.handlers import RotatingFileHandler
from os import makedirs
from os.path import join, dirname
from sys import stdout

from model.config import PROJECT_NAME
from model.run import run


def init_logger(args):
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("torchvision").setLevel(logging.ERROR)
    log = logging.getLogger('')
    log.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(stdout)
    ch.setFormatter(formatter)
    log.addHandler(ch)

    log_file_path = join("logs", PROJECT_NAME, '_'.join((args['model_key'], args['data_key'] + ".log")))
    makedirs(dirname(log_file_path), exist_ok=True)
    fh = RotatingFileHandler(log_file_path + "", maxBytes=(1048576 * 5), backupCount=7)
    fh.setFormatter(formatter)
    log.addHandler(fh)


def main():
    parser = ArgumentParser()
    action_parser = parser.add_subparsers(title="actions", dest="action", required=True,
                                          help="select action to execute")

    # args for training
    train_parser = action_parser.add_parser("train", help="train the classifier")
    train_parser.add_argument("-r", "--root-dir", dest="root_dir", required=True,
                              help="root directory of the dataset")
    train_parser.add_argument("-d", "--data-key", dest="data_key", required=True,
                              help="name of the dataset")
    train_parser.add_argument("-m", "--model-name", dest="model_key", required=True,
                              help="model to be used for training")
    train_parser.add_argument("-s", "--save-path", dest="save_path", required=True,
                              help="save path for trained model")

    # args for testing
    test_parser = action_parser.add_parser("test", help="test the classifier")
    test_parser.add_argument("-r", "--root-dir", dest="root_dir", required=True,
                             help="root directory of the dataset")
    test_parser.add_argument("-d", "--data-key", dest="data_key", required=True,
                             help="name of the dataset")
    test_parser.add_argument("-m", "--model-name", dest="model_key", required=True,
                             help="model to be used for testing")
    test_parser.add_argument("-s", "--save-path", dest="save_path", required=True,
                             help="save path of the trained model")

    args_dict = vars(parser.parse_args())
    init_logger(args_dict)
    logging.info(f'User Arguments: {args_dict}!!!')
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
