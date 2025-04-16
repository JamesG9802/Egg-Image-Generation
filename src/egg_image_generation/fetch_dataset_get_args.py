from argparse import ArgumentParser, Namespace

def get_args() -> Namespace:
    """Returns the command line arguments for `fetch_dataset`.

    Returns:
        Namespace: the arguments.
    """
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument(
        "target",
        type=str,
        help="Where the files will be downloaded to."
    )

    return parser.parse_args()

