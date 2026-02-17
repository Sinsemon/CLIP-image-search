from const import TIMING
from utils import catch_time

from pathlib import Path
from csv import reader
from typing import Self
from torch import load, Tensor, save
import torch


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


    def __init__(self, save_path:str|Path) -> None:
        self.db_save_path = Path(save_path).absolute()
        self.ids = []
        self.img_paths = []
        self.img_embeddings = []


    def __iter__(self):
        for _id, _img_path, _img_embedding in zip(self.ids, self.img_paths, self.img_embeddings):
            yield _id, _img_path, _img_embedding


    def load(self) -> Self:
        """
        Load data from existing database
        
        :return: Returns self. Therefore it is chainable with the constructor.
        :rtype: Self
        """
        with catch_time(TIMING, "Database load"):
            with open(self.db_save_path / "db.csv", mode="r", newline="") as f:
                _root, _id = f.readline().strip().split(",")
                self._current_id = int(_id)
                if _root != "None":
                    self.db_root = Path(_root)
                _csv_reader = reader(f)
                for row in _csv_reader:
                    self.ids.append(int(row[0]))
                    self.img_paths.append(Path(row[1]))
                    self.img_embeddings.append(load(self.db_save_path / "tensors" / f"{row[0]}.pt"))
        return self


    def add(self, img_path:str|Path, embedding:Tensor) -> None:
        _path = Path(img_path)
        if _path.is_file():
            self.ids.append(self._current_id)
            self._current_id += 1
            self.img_paths.append(Path(img_path))
            self.img_embeddings.append(embedding)
        else:
            raise Exception(f"Image path {img_path} is not a file.")


    def get_by_id(self, img_id:int) -> tuple[Path, Tensor]:
        """Get an image embedding by id"""
        _index = self.ids.index(img_id)
        return self.img_paths[_index], self.img_embeddings[_index]
    

    def get_similar(self, embedding:Tensor, n:int=20) -> list[tuple[float, Path]]:
        """
        Get the `n` most similar images from the database using the cosine similarity
        
        :param embedding: Reference embedding to which the most similar database entries are searched
        :type embedding: Tensor
        :param n: How many entries should be returned
        :type n: int
        :return: A list of the most similar entries (similarity score, Path to image) sorted in descending order.
        :rtype: list[tuple[float, Path]]
        """
        with catch_time(TIMING, "Database get_similar"):
            _embeddings_img = torch.cat(self.img_embeddings)
            similarities = torch.nn.functional.cosine_similarity(embedding, _embeddings_img)
            indizes = torch.argsort(similarities).__reversed__()
        return [(similarities[index].item(), self.img_paths[index]) for index in indizes.cpu().numpy()[:n]]


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
            raise NotImplementedError() # TODO
        else:
            with open(self.db_save_path / "db.csv", "x") as db_file:
                db_file.write(f'{'None' if self.db_root is None else self.db_root.absolute()},{self._current_id}\n')
                _tensors_path = self.db_save_path / "tensors"
                _tensors_path.mkdir()
                for _id, _img_path, _img_embedding in zip(self.ids, self.img_paths, self.img_embeddings):
                    db_file.write(f"{_id},{_img_path.absolute()}\n")
                    save(_img_embedding, (_tensors_path / f"{_id}.pt"))


    def embed_all(self, root_dir:str) -> None:
        """
        Find all images and embed them. Fill this database with the embeddings.
        
        :param root_dir: Root directory under which all images will be embedded.
        :type root_dir: str
        """
        self.db_root = Path(root_dir).absolute()
        raise NotImplementedError() # TODO

    def clean(self) -> None:

        if self.db_root is None:
            raise Exception("The 'db_root' must exist. Open an existing database first that has a 'db_root'.")
        raise NotImplementedError() # TODO