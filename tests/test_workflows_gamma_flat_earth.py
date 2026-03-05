from pathlib import Path

from dolphin.workflows import wrapped_phase


def _write_gamma_inputs(tmp_path: Path) -> list[Path]:
    inputs = []
    for d in ["20200101", "20200113"]:
        slc = tmp_path / f"{d}.rslc"
        slc.write_bytes(b"\x00" * 16)
        par = tmp_path / f"{d}.rslc.par"
        par.write_text(
            "range_samples: 2\nazimuth_lines: 2\nimage_format: SCOMPLEX\n",
            encoding="utf-8",
        )
        vrt = tmp_path / f"{d}.rslc.vrt"
        vrt.write_text(
            "\n".join(
                [
                    '<VRTDataset rasterXSize="2" rasterYSize="2">',
                    (
                        '  <VRTRasterBand dataType="CInt16" band="1"'
                        ' subClass="VRTRawRasterBand">'
                    ),
                    '    <SourceFilename relativeToVRT="1">'
                    + slc.name
                    + "</SourceFilename>",
                    "    <ImageOffset>0</ImageOffset>",
                    "    <PixelOffset>4</PixelOffset>",
                    "    <LineOffset>8</LineOffset>",
                    "    <ByteOrder>MSB</ByteOrder>",
                    "  </VRTRasterBand>",
                    "</VRTDataset>",
                ]
            ),
            encoding="utf-8",
        )
        inputs.append(vrt)
    return inputs


def test_parse_gamma_slc_inputs(tmp_path):
    inputs = _write_gamma_inputs(tmp_path)
    out = wrapped_phase._parse_gamma_slc_inputs(inputs, "%Y%m%d")

    assert len(out) == 2
    assert out[0].slc_par.exists()
    assert out[0].slc_bin.suffix == ".rslc"


def test_maybe_generate_gamma_sim_orb(tmp_path, monkeypatch):
    inputs = _write_gamma_inputs(tmp_path)
    hgt = tmp_path / "dem.rdc"
    hgt.write_bytes(b"\x00" * 16)

    monkeypatch.setattr(wrapped_phase.shutil, "which", lambda _cmd: "/usr/bin/fake")

    def _fake_run(cmd, check, stdout, stderr):
        assert check is True
        # create .off output
        if cmd[0] == "create_offset":
            Path(cmd[3]).write_text("offset", encoding="utf-8")
        # create sim_orb output
        elif cmd[0] == "phase_sim_orb":
            Path(cmd[5]).write_bytes(b"\x00" * 16)
        return 0

    monkeypatch.setattr(wrapped_phase.subprocess, "run", _fake_run)

    out = wrapped_phase._maybe_generate_gamma_sim_orb(
        input_file_list=inputs,
        date_fmt="%Y%m%d",
        hgt_file=hgt,
        reference_date="20200101",
        work_directory=tmp_path,
        scratch_dir=tmp_path / "scratch",
    )

    assert len(out) == 2
    for vrt in out:
        assert vrt.exists()
        assert vrt.read_text(encoding="utf-8").find("VRTRawRasterBand") > 0


def test_maybe_generate_gamma_sim_orb_skip_when_complete(tmp_path, monkeypatch):
    inputs = _write_gamma_inputs(tmp_path)
    hgt = tmp_path / "dem.rdc"
    hgt.write_bytes(b"\x00" * 16)

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    # 2x2 float32 => 16 bytes expected
    for d in ["20200101", "20200113"]:
        (scratch / f"20200101_{d}.sim_orb").write_bytes(b"\x00" * 16)

    monkeypatch.setattr(wrapped_phase.shutil, "which", lambda _cmd: "/usr/bin/fake")

    def _fail_run(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be called when sim_orb exists")

    monkeypatch.setattr(wrapped_phase.subprocess, "run", _fail_run)

    out = wrapped_phase._maybe_generate_gamma_sim_orb(
        input_file_list=inputs,
        date_fmt="%Y%m%d",
        hgt_file=hgt,
        reference_date="20200101",
        work_directory=tmp_path,
        scratch_dir=scratch,
    )

    assert len(out) == 2
    for vrt in out:
        assert vrt.exists()
