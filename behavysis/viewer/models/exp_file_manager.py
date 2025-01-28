import os


class ExpFileManager:
    _root_dir: str
    _name: str

    def __init__(self, *args, **kwargs):
        pass

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def name(self):
        return self._name

    @property
    def vid_fp(self):
        return os.path.join(self.root_dir, "2_formatted_vid", f"{self.name}.mp4")

    @property
    def behavs_df_fp(self):
        return os.path.join(self.root_dir, "7_scored_behavs", f"{self.name}.feather")

    @property
    def dlc_df_fp(self):
        return os.path.join(self.root_dir, "4_preprocessed", f"{self.name}.feather")

    @property
    def configs_fp(self):
        return os.path.join(self.root_dir, "0_configs", f"{self.name}.json")

    @root_dir.setter
    def root_dir(self, value: str):
        self._root_dir = value

    @name.setter
    def name(self, value: str):
        self._name = value

    def load(self, fp: str):
        self.root_dir = os.path.split(os.path.split(fp)[0])[0]
        self.name = os.path.splitext(os.path.split(fp)[1])[0]


if __name__ == "__main__":
    vfm = ExpFileManager()
