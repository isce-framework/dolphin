import threading
import time
from pathlib import Path

from dolphin._decorators import atomic_output


def _long_write(filename, pause: float = 0.2):
    """Simulate a long writing process"""
    with open(filename, "w") as f:
        f.write("aaa\n")
        time.sleep(pause)
        f.write("bbb\n")


@atomic_output(output_arg="outname")
def default_write_newname(outname="out.txt"):
    _long_write(outname)


@atomic_output(output_arg="outname", use_tmp=True)
def default_write_newname_tmp(outname="out.txt"):
    _long_write(outname)


@atomic_output(output_arg="output_dir", is_dir=True)
def default_write_dir(output_dir="some_dir", filename="testfile.txt"):
    p = Path(output_dir)
    p.mkdir(exist_ok=True, parents=True)
    outname = p / filename
    _long_write(outname)


@atomic_output(output_arg="output_dir", is_dir=True, use_tmp=True)
def default_write_dir_tmp(output_dir="some_dir", filename="testfile.txt"):
    p = Path(output_dir)
    p.mkdir(exist_ok=True, parents=True)
    outname = p / filename
    _long_write(outname)


def test_atomic_output(tmpdir):
    with tmpdir.as_cwd():
        default_write_newname(outname="out1.txt")
        default_write_newname(outname="out2.txt")
        for fn in ["out1.txt", "out2.txt"]:
            assert Path(fn).exists()


def test_atomic_output_tmp(tmpdir):
    with tmpdir.as_cwd():
        default_write_newname_tmp(outname="out1.txt")
        assert Path("out1.txt").exists()


def test_atomic_output_dir(tmp_path: Path):
    out_dir = tmp_path / "out"
    filename = "testfile.txt"
    out_dir.mkdir()
    default_write_dir(output_dir=out_dir, filename=filename)
    assert Path(out_dir / filename).exists()


def test_atomic_output_dir_tmp(tmp_path: Path):
    out_dir = tmp_path / "out"
    filename = "testfile.txt"
    out_dir.mkdir()
    default_write_dir(output_dir=out_dir, filename=filename)
    assert Path(out_dir / filename).exists()


def test_atomic_output_name_swap_file(tmpdir):
    with tmpdir.as_cwd():
        outname2 = "out3.txt"
        t = threading.Thread(target=default_write_newname, kwargs={"outname": outname2})
        t.start()
        # It should NOT exist, yet
        assert not Path(outname2).exists()
        time.sleep(0.5)
        t.join()
        assert Path(outname2).exists()


def test_atomic_output_dir_swap(tmp_path: Path):
    # Kick off the writing function in the background
    # so we see if a different file was created
    # Check it works providing the "args"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    t = threading.Thread(target=default_write_dir, kwargs={"output_dir": out_dir})
    t.start()
    # It should NOT exist, yet
    assert not Path(out_dir / "testfile.txt").exists()
    time.sleep(0.5)
    t.join()
    assert Path(out_dir / "testfile.txt").exists()
