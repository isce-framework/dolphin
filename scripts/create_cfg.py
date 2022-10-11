#!/usr/bin/env python
import fire


def fill_cfg(
    slc_file_path,
    slc_file_ext: str = ".slc",
    num_slcs: int = 5,
    output: str = "runconfig.yaml",
    scratch_path: str = "scratch",
    xhalf: int = None,
    yhalf: int = None,
    amp_disp_thresh: float = None,
    pl_method: str = None,
    nmap_pvalue: float = None,
):
    """Create a runconfig.yaml file for the disp_s1_workflow.

    This is a convenience function to create a runconfig.yaml file for the
    workflow, where only the specified options are filled (along with the
    necessary PGE options).
    """
    filled = f"""runconfig:
  name: disp_s1_workflow_default

  groups:
    input_file_group:
      cslc_file_path: "{slc_file_path}"
      cslc_file_ext: "{slc_file_ext}"

    product_path_group:
      product_path: 'output'
      scratch_path: "{scratch_path}"
      sas_output_file: 'unwrapped_final.unw'


    processing:
    """
    if xhalf is not None and yhalf is not None:
        print(f"{xhalf = } and {yhalf = }")
        filled += f"""
      window:
        xhalf: {xhalf}
        yhalf: {yhalf}
"""
    if amp_disp_thresh is not None:
        print(f"{amp_disp_thresh = }")
        filled += f"""
      ps:
        amp_dispersion_threshold: {amp_disp_thresh}
"""
    if nmap_pvalue is not None:
        print(f"{nmap_pvalue = }")
        filled += f"""
      nmap:
        pvalue: {nmap_pvalue}
        stat_method: 'KS2'

"""
    if pl_method is not None:
        print(f"{pl_method = }")
        filled += f"""
      phase_link:
        minimum_neighbors: 5
        method: {pl_method}
"""
    lines = filled.splitlines()
    if not lines[-1].strip():
        lines = lines[:-1]
    if lines[-1].strip() == "processing:":
        filled = "\n".join(lines[:-1])
    print("Writing config file to", output)
    with open(output, "w") as f:
        f.write(filled)


# unwrap:
# unwrap_method: 'snaphu'
# tiles: [1, 1]


if __name__ == "__main__":
    # https://github.com/google/python-fire/issues/188#issuecomment-791972163
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(fill_cfg)
