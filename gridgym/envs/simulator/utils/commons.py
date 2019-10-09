import os
import shutil
import sys

def overwrite_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def signal_wrapper(call):
    def cleanup(signum, frame):
        call()
        sys.exit(signum)
    assert callable(call)
    return cleanup