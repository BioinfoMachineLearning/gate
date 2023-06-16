import os, sys, argparse
import contextlib
import shutil
import tempfile
import time
from typing import Optional
from absl import logging

def makedir_if_not_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.abspath(directory)
    return directory

@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None):
  """Context manager that deletes a temporary directory on exit."""
  tmpdir = tempfile.mkdtemp(dir=base_dir)
  try:
    yield tmpdir
  finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


@contextlib.contextmanager
def timing(msg: str):
  logging.info('Started %s', msg)
  tic = time.time()
  yield
  toc = time.time()
  logging.info('Finished %s in %.3f seconds', msg, toc - tic)
