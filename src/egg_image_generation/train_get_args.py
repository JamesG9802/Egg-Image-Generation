from argparse import ArgumentParser, Namespace

def get_args() -> Namespace:
    """Returns the command line arguments for `train`.

    Returns:
        Namespace: the arguments.
    """
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument(
        "source",
        type=str,
        help="The source directory to read from. The dataset folder should have 'Damaged/' and 'Not Damaged/' folders."
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="The random seed."
    )

    parser.add_argument(
        "-i",
        "--image_size",
        type=int,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        default=(96, 128),
        help="The dimensions of each image."
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="The size of the batch."
    )
    
    parser.add_argument(
        "-z",
        type=int,
        default=100,
        help="The size of the random noise channel."
    )

    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=1e-3,
        help="The starting learning rate for the generator and discriminator."
    )

    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=100,
        help="The number of epochs to train on."
    )

    return parser.parse_args()

