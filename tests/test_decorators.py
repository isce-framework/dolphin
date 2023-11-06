import threading
import time
from pathlib import Path

from dolphin._decorators import atomic_output


def _long_write(filename, pause: float = 0.2):
    """Simulate a long writing process"""
    f = open(filename, "w")
    f.write("aaa\n")
    time.sleep(pause)
    f.write("bbb\n")
    f.close()


@atomic_output
def default_write(output_file="out.txt"):
    _long_write(output_file)


@atomic_output(output_arg="outname")
def default_write_newname(outname="out3.txt"):
    _long_write(outname)


@atomic_output(output_arg="output_dir")
def default_write_dir(output_dir="some_dir"):
    p = Path(output_dir)
    p.mkdir(exist_ok=True)
    outname = p / "testfile.txt"
    _long_write(outname)


def test_atomic_output(tmpdir):
    with tmpdir.as_cwd():
        default_write()
        default_write(output_file="out2.txt")
        default_write_newname()
        default_write_newname(outname="out4.txt")
        for fn in ["out.txt", "out2.txt", "out3.txt", "out4.txt"]:
            assert Path(fn).exists()


def test_atomic_output_name_swap(tmpdir):
    # Kick off the writing function in the background
    # so we see if a different file was created
    with tmpdir.as_cwd():
        # Check it works providing the "args"
        t = threading.Thread(target=default_write)
        t.start()
        # It should NOT exist, yet
        assert not Path("out.txt").exists()
        time.sleep(0.5)
        assert Path("out.txt").exists()
        Path("out.txt").unlink()


def test_atomic_output_name_swap_with_args(tmpdir):
    with tmpdir.as_cwd():
        outname2 = "out2.txt"
        t = threading.Thread(target=default_write, args=(outname2,))
        t.start()
        # It should NOT exist, yet
        assert not Path(outname2).exists()
        time.sleep(0.5)
        t.join()
        assert Path(outname2).exists()
        Path(outname2).unlink()


def test_atomic_output_name_swap_with_kwargs(tmpdir):
    with tmpdir.as_cwd():
        outname2 = "out3.txt"
        t = threading.Thread(target=default_write_newname, kwargs={"outname": outname2})
        t.start()
        # It should NOT exist, yet
        assert not Path(outname2).exists()
        time.sleep(0.5)
        t.join()
        assert Path(outname2).exists()
        Path(outname2).unlink()


def test_atomic_output_dir_name_swap(tmpdir):
    # Kick off the writing function in the background
    # so we see if a different file was created
    with tmpdir.as_cwd():
        # Check it works providing the "args"
        t = threading.Thread(target=default_write)
        t.start()
        # It should NOT exist, yet
        assert not Path("out.txt").exists()
        time.sleep(0.5)
        assert Path("out.txt").exists()
        Path("out.txt").unlink()
