from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import tyro
from pydantic import Field, model_validator

from dolphin.workflows import DisplacementWorkflow


class ConfigCli(DisplacementWorkflow):
    print_empty: bool = Field(
        False,
        description="Print an empty YAML file with only default filled to `outfile`.",
    )
    outfile: Path = Field(
        description="Output file for the configuration.",
        default=Path("dolphin_config.yaml"),
    )

    @model_validator(mode="before")
    @classmethod
    def short_circuit_validation(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("print_empty", False):
            DisplacementWorkflow.print_yaml_schema(
                data.get("outfile", "dolphin_config.yaml")
            )
            data["require_cslc_files"] = False
            return data
        return data


if __name__ == "__main__":
    cfg = tyro.cli(ConfigCli)
    print(f"Saving configuration to {cfg.outfile!s}", file=sys.stderr)
    cfg.to_yaml(cfg.outfile)
