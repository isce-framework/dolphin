from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from dolphin.workflows import DisplacementWorkflow


class ConfigCli(DisplacementWorkflow):
    """Create a configuration file for a displacement workflow."""

    print_empty: bool = Field(
        False,
        description="Print an empty YAML file with only default filled to `outfile`.",
        # hide these from the output dump
        exclude=True,
    )
    outfile: Path = Field(
        description="Output file for the configuration.",
        default=Path("dolphin_config.yaml"),
        exclude=True,
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

    @model_validator(mode="after")
    def save_config(self) -> Any:
        print(f"Saving configuration to {self.outfile!s}", file=sys.stderr)
        self.to_yaml(self.outfile)
        return self
