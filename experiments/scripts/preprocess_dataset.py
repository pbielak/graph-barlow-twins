import sys

from gssl.datasets import load_dataset
from gssl.inductive.datasets import load_ppi
from gssl.utils import seed


def main():
    seed()

    # Read dataset name
    dataset_name = sys.argv[1]

    # Load dataset
    if dataset_name == "PPI":
        load_ppi()
    else:
        load_dataset(name=dataset_name)


if __name__ == "__main__":
    main()
