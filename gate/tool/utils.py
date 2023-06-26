import os, sys, argparse
from typing import Optional

def makedir_if_not_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.abspath(directory)
    return directory

