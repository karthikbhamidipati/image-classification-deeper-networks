import logging
from argparse import ArgumentParser

from model.run import run


def init_logger():
    logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                      datefmt='%Y-%m-%d %H:%M:%S')


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
    init_logger()
    logging.debug(f'User Arguments: {args_dict}!!!')
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
