import os, sys, argparse
import contextlib
import shutil
import tempfile
import time
from typing import Optional


def die(msg):
    print(msg)
    sys.exit(1)


def check_dirs(params, keys, isdir=True):
    errmsg = ''
    for key in keys:
        dirpath = params[key]
        # print(f"{key}:{params[key]}")
        if isdir and not os.path.isdir(dirpath):
            errmsg = errmsg + '{}({})\n'.format(key, dirpath)

        if not isdir and not os.path.exists(dirpath):
            errmsg = errmsg + '{}({})\n'.format(key, dirpath)

    if len(errmsg) > 0:
        errmsg = 'Directories or files are not exist:\n' + errmsg
        raise argparse.ArgumentTypeError(errmsg)


def check_contents(params, keys):
    errmsg = ''
    for key in keys:
        name = params[key]
        if len(name) == 0:
            errmsg = errmsg + '{}\n'.format(key)

    if len(errmsg) > 0:
        errmsg = 'These contents are emply:\n' + errmsg
        raise argparse.ArgumentTypeError(msg)


def is_dir(dirname):
    """Checks if a path is an actual directory"""
    if not os.path.isdir(dirname):
        msg = "{0} is not a directory".format(dirname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dirname


def check_dir(dirname):
    return is_dir(dirname)


def is_file(filename):
    """Checks if a file is an invalid file"""
    if not os.path.exists(filename):
        msg = "{0} doesn't exist".format(filename)
        raise argparse.ArgumentTypeError(msg)
    else:
        return filename


def check_file(dirname):
    return is_file(dirname)


def makedir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.abspath(directory)
    return directory


def run_command(cmd, log_file=None):
    flag = os.system(cmd)
    errmsg = 'Error occur while run: %s' % cmd
    if flag:
        print(errmsg)
        if log_file != None:
            write_str2file(log_file, errmsg)


def clean_dir(dir):
    if os.path.exists(dir):
        os.system(f'rm -rf {dir}')
    os.makedirs(dir)
