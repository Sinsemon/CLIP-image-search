from pathlib import Path
from csv import reader
from typing import Self
from torch import load, Tensor


class Database:
    """
    Database # TODO
    """
    _current_id = 0  # smallest valid id

    db_save_path:Path
    db_root:Path|None = None
    ids:list[int]
    img_paths:list[Path]
    img_embeddings:list[Tensor]


    def __init__(self, save_path:str) -> None:
        self.db_save_path = Path(save_path)
        self.ids = []
        self.img_paths = []
        self.img_embeddings = []

    def load(self) -> Self:
        """
        Load data from existing database
        
        :return: Returns self. Therefore it is chainable with the constructor.
        :rtype: Self
        """
        with open(self.db_save_path, mode="r", newline="") as f:
            _root, _id = f.readline().strip().split(",")
            self._current_id = int(_id)
            if _root != "None":
                self.db_root = Path(_root)
            _csv_reader = reader(f)
            for row in _csv_reader:
                self.ids.append(int(row[0]))
                self.img_paths.append(Path(row[1]))
                self.img_embeddings.append(load(self.db_save_path / f"{row[0]}.pt"))
        return self

    def add(self, img_path, embedding) -> None:
        pass # TODO

    def get(self, img_id:int) -> tuple[Path, Tensor]:
        """Get an image embedding by id"""
        _index = self.ids.index(img_id)
        return self.img_paths[_index], self.img_embeddings[_index]

    def save(self) -> None:
        """
        Save database to disk or append new items to an existing database.
        
        Storage model of `db.csv` file:

        ```
        1 <db_root path>, _current_id
        2 <id>, <img path>
        3 <id>, <img path>
        ...
        ```

        Disk layout under `db_save_path/`:

        ```
        tensors/
        db.csv
        ```

        The corresponding tensor can be found under `tensors/<id>.pt`
        """
        if Path.is_file(self.db_save_path / "db.csv"):
            pass # TODO
        else:
            with open(self.db_save_path / "db.csv", "x") as db_file:
                db_file.write(f'{'None' if self.db_root is None else self.db_root.absolute()},{self._current_id}')
                # TODO create tensors dir
                # TODO write tensors in dir


    def embed_all(self, root_dir:str) -> None:
        """
        Find all images and embed them
        
        :param root_dir: Root directory under which all images will be embedded.
        :type root_dir: str
        """
        self.db_root = Path(root_dir)
        pass # TODO

    def clean(self, only_save=False) -> None:

        if self.db_root is None:
            raise Exception("The 'db_root' must exist. Open an existing database first that has a 'db_root'.")
        pass # TODO