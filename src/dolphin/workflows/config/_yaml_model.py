from __future__ import annotations

import json
import sys
import textwrap
from io import StringIO
from typing import Optional, TextIO, Union

from pydantic import BaseModel
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from dolphin._types import Filename


class YamlModel(BaseModel):
    """Pydantic model that can be exported to yaml."""

    def to_yaml(
        self,
        output_path: Union[Filename, TextIO],
        with_comments: bool = True,
        by_alias: bool = True,
        indent_per_level: int = 2,
    ):
        """Save configuration as a yaml file.

        Used to record the default-filled version of a supplied yaml.

        Parameters
        ----------
        output_path : Pathlike
            Path to the yaml file to save.
        with_comments : bool, default = False.
            Whether to add comments containing the type/descriptions to all fields.
        by_alias : bool, default = False.
            Whether to use the alias names for the fields.
            Passed to pydantic's ``to_json`` method.
            https://docs.pydantic.dev/usage/exporting_models/#modeljson
        indent_per_level : int, default = 2
            Number of spaces to indent per level.

        """
        yaml_obj = self._to_yaml_obj(by_alias=by_alias)

        if with_comments:
            _add_comments(
                yaml_obj,
                self.model_json_schema(by_alias=by_alias),
                indent_per_level=indent_per_level,
            )

        y = YAML()
        # https://yaml.readthedocs.io/en/latest/detail.html#indentation-of-block-sequences
        y.indent(
            offset=indent_per_level,
            mapping=indent_per_level,
            # It is best to always have sequence >= offset + 2 but this is not enforced
            # not following this advice might lead to invalid output.
            sequence=indent_per_level + 2,
        )
        if hasattr(output_path, "write"):
            y.dump(yaml_obj, output_path)
        else:
            with open(output_path, "w") as f:
                y.dump(yaml_obj, f)

    @classmethod
    def from_yaml(cls, yaml_path: Filename):
        """Load a configuration from a yaml file.

        Parameters
        ----------
        yaml_path : Pathlike
            Path to the yaml file to load.

        Returns
        -------
        Config
            Workflow configuration

        """
        y = YAML(typ="safe")
        with open(yaml_path) as f:
            data = y.load(f)

        return cls(**data)

    @classmethod
    def print_yaml_schema(
        cls,
        output_path: Union[Filename, TextIO] = sys.stdout,
        indent_per_level: int = 2,
    ):
        """Print/save an empty configuration with defaults filled in.

        Ignores the required `cslc_file_list` input, so a user can
        inspect all fields.

        Parameters
        ----------
        output_path : Pathlike
            Path or stream to save to the yaml file to.
            By default, prints to stdout.
        indent_per_level : int, default = 2
            Number of spaces to indent per level.

        """
        cls.model_construct().to_yaml(
            output_path, with_comments=True, indent_per_level=indent_per_level
        )

    def _to_yaml_obj(self, by_alias: bool = True) -> CommentedMap:
        # Make the YAML object to add comments to
        # We can't just do `dumps` for some reason, need a stream
        y = YAML()
        ss = StringIO()
        y.dump(json.loads(self.model_dump_json(by_alias=by_alias)), ss)
        return y.load(ss.getvalue())


def _add_comments(
    loaded_yaml: CommentedMap,
    schema: dict,
    indent: int = 0,
    definitions: Optional[dict] = None,
    # variable specifying how much to indent per level
    indent_per_level: int = 2,
):
    """Add comments above each YAML field using the pydantic model schema."""
    # Definitions are in schemas that contain nested pydantic Models
    defs = schema.get("$defs") if definitions is None else definitions

    for key, val in schema["properties"].items():
        reference = ""
        # Get sub-schema if it exists
        if "$ref" in val:
            # At top level, example is 'outputs': {'$ref': '#/defs/Outputs'}
            reference = val["$ref"]
        elif "allOf" in val:
            # within 'defs', it looks like
            #  'allOf': [{'$ref': '#/defs/HalfWindow'}]
            reference = val["allOf"][0]["$ref"]

        ref_key = reference.split("/")[-1]
        if ref_key:  # The current property is a reference to something else
            if "enum" in defs[ref_key]:  # type: ignore[index]
                # This is just an Enum, not a sub schema.
                # Overwrite the value with the referenced value
                val = defs[ref_key]  # type: ignore[index] # noqa: PLW2901
            else:
                # The reference is a sub schema, so we need to recurse
                sub_schema = defs[ref_key]  # type: ignore[index]
                # Get the sub-model
                sub_loaded_yaml = loaded_yaml[key]
                # recurse on the sub-model
                _add_comments(
                    sub_loaded_yaml,
                    sub_schema,
                    indent=indent + indent_per_level,
                    definitions=defs,
                    indent_per_level=indent_per_level,
                )
                continue

        # add each description along with the type information
        desc = "\n".join(
            textwrap.wrap(
                f"{val['description']}.",
                width=90,
                subsequent_indent=" " * indent_per_level,
            )
        )
        if "anyOf" in val:
            #   'anyOf': [{'type': 'string'}, {'type': 'null'}],
            # Join the options with a pipe, like Python types
            type_str = " | ".join(d["type"] for d in val["anyOf"])
            type_str.replace("null", "None")
        elif "const" in val:
            type_str = val["const"]
        else:
            type_str = val["type"]
        type_line = f"\n  Type: {type_str}."
        choices = f"\n  Options: {val['enum']}." if "enum" in val else ""

        # Combine the description/type/choices as the YAML comment
        comment = f"{desc}{type_line}{choices}"
        comment = comment.replace("..", ".")  # Remove double periods

        # Prepend the required label for fields that are required
        is_required = key in schema.get("required", [])
        if is_required:
            comment = "REQUIRED: " + comment

        # This method comes from here
        # https://yaml.readthedocs.io/en/latest/detail.html#round-trip-including-comments
        loaded_yaml.yaml_set_comment_before_after_key(key, comment, indent=indent)
