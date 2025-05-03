from argparse import ArgumentParser, Namespace

def get_args() -> Namespace:
    """Returns the command line arguments for `view`.

    Returns:
        Namespace: the arguments.
    """
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        help="The model containing the generator to load."
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
        "-z",
        type=int,
        default=100,
        help="The size of the random noise channel."
    )

    return parser.parse_args()

