import errno
import os


def mkdir_p(path):
    """Emulates bash `mkdir -p`, in python style."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
