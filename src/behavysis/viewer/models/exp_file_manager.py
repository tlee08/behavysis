from pathlib import Path

from behavysis.constants import DF_IO_FORMAT


class ExpFileManager:
    _root_dir: Path
    _name: str

    def __init__(self, *args, **kwargs):
        pass

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def name(self) -> str:
        return self._name

    @property
    def vid_fp(self) -> Path:
        return self.root_dir / "2_formatted_vid" / f"{self.name}.mp4"

    @property
    def behavs_df_fp(self) -> Path:
        return self.root_dir / "7_scored_behavs" / f"{self.name}.{DF_IO_FORMAT}"

    @property
    def dlc_df_fp(self) -> Path:
        return self.root_dir / "4_preprocessed" / f"{self.name}.{DF_IO_FORMAT}"

    @property
    def configs_fp(self) -> Path:
        return self.root_dir / "0_configs" / f"{self.name}.json"

    @root_dir.setter
    def root_dir(self, value: Path) -> None:
        self._root_dir = value

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def load(self, fp: Path):
        self.root_dir = fp.parent.parent
        self.name = fp.stem


if __name__ == "__main__":
    vfm = ExpFileManager()
