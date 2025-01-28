"""
Utility functions.
"""

import os

from pydantic import BaseModel, ConfigDict


class PydanticBaseModel(BaseModel):
    """Helper class for Pydantic models (i.e. configs)."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def read_json(cls, fp: str):
        """
        Returns the config model from the specified JSON config file.

        Parameters
        ----------
        fp : str
            Filepath of the JSON config file.

        Notes
        -----
        This class method reads the contents of the JSON config file located at `fp` and
        returns the config model.

        Example
        -------
        >>> config = ConfigModel.read_json("/path/to/config.json")
        """
        with open(fp, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    def write_json(self, fp: str) -> None:
        """
        Writes the given configs model to the configs file (i.e. hence updating the file).

        Makes the directory if it doesn't exist.

        Parameters
        ----------
        fp : str
            File to save configs to.
        """
        fp_dir = os.path.dirname(fp)
        os.makedirs(fp_dir, exist_ok=True) if fp_dir else None
        with open(fp, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @staticmethod
    def validate_attrs(model, field_names, model_cls):
        """Convert the attributes of the model to the correct type."""
        for k in field_names:
            try:
                v = getattr(model, k)
                setattr(model, k, model_cls.model_validate(v))
            except Exception as e:
                raise ValueError(f"'{k}' is not a valid field name") from e
        return model

    @staticmethod
    def validate_attr_closed_set(v, closed_set):
        """Validate that the attribute is in the given closed set."""
        if v not in closed_set:
            raise ValueError(f"Invalid value: {v}.\nOption must be one of: {', '.join(closed_set)}")
        return v

    @classmethod
    def get_field_names(cls) -> list[tuple[str, ...]]:
        """
        Returns the nested field names of the class as
        a list of tuples.
        Each tuple is a nested field name combination.

        For example, the following
        ```
        {
            "field1": {
                "fieldA": xxx,
                "fieldB": {
                    "fieldI": xxx,
                    "fieldII": xxx,
                },
            },
            "field2": xxx,
        }
        ```
        Becomes
        ```
        [
            ("field1", "fieldA"),
            ("field1", "fieldB", "fieldI"),
            ("field1", "fieldB", "fieldII"),
            ("field2",),
        ]
        ```
        """
        fields = []
        # For each field in the class
        for name, type_ in cls.__annotations__.items():
            if hasattr(type_, "__annotations__"):  # Is a nested class
                for subfield in type_.get_field_names():
                    fields.append((name,) + subfield)
            else:  # is a primitive field
                fields.append((name,))
        return fields
