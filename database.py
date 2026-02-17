from const import TIMING
from utils import catch_time
from clip_model import Clip

from pathlib import Path
from csv import reader
from typing import Self
from PIL import Image
from torch import load, Tensor
import torch


class Database:
    """
    Database # TODO
    """
    _current_id = 0  # smallest valid id

    db_save_path:Path
    img_root:Path|None = None

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


    def _recurse_images(self, path:Path):
        image_endings = [".jpg", ".png", ".jpeg"]
        for f in path.rglob("*"):
            if f.is_file():
                if f.suffix.lower() in image_endings:
                    yield f

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
                    self.img_root = Path(_root)
                _csv_reader = reader(f)
                for row in _csv_reader:
                    self.ids.append(int(row[0]))
                    self.img_paths.append(Path(row[1]))
                    self.img_embeddings.append(load(self.db_save_path / "tensors" / f"{row[0]}.pt"))
        return self


    def add(self, img_path:str|Path, embedding:Tensor) -> int:
        _path = Path(img_path)
        if _path.is_file():
            self.ids.append(self._current_id)
            self._current_id += 1
            self.img_paths.append(Path(img_path))
            self.img_embeddings.append(embedding)
            return self._current_id - 1
        else:
            raise Exception(f"Image path {img_path} is not a file.")


    def remove(self, img_path:Path) -> int:
        """
        Remove a given image from the database.
        
        :param img_path: Path to the image.
        :type img_path: Path
        :return: Returns the ID of the removed image.
        :rtype: int
        """
        _index = self.img_paths.index(img_path)
        self.img_embeddings.pop(_index)
        self.img_paths.pop(_index)
        return self.ids.pop(_index)
    

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
        Save a new database to disk.
        
        Storage model of `db.csv` file:

        ```
        1 <img_root path>, _current_id
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
            raise DatabaseExists(self.db_save_path)
        else:
            with catch_time(TIMING, "Database save"):
                with open(self.db_save_path / "db.csv", "x") as db_file:
                    # db_file.write(f'{'None' if self.img_root is None else self.img_root.absolute()},{self._current_id}\n')
                    self._write_dbcsv_header(db_file)
                    # _tensors_path = self.db_save_path / "tensors"
                    # _tensors_path.mkdir()
                    _tensors_path = self._make_tensors_dir()

                    for _id, _img_path, _img_embedding in zip(self.ids, self.img_paths, self.img_embeddings):
                        # db_file.write(f"{_id},{_img_path.absolute()}\n")
                        # torch.save(_img_embedding, (_tensors_path / f"{_id}.pt"))
                        self._save_item(db_file, _tensors_path, _id, _img_path, _img_embedding)


    def _write_dbcsv_header(self, file):
        file.write(f'{'None' if self.img_root is None else self.img_root.absolute()},{self._current_id}\n')
    
    def _make_tensors_dir(self):
        _tensors_path = self.db_save_path / "tensors"
        _tensors_path.mkdir()
        return _tensors_path
    
    def _save_item(self, file, tensors_path, _id, _img_path, _img_embedding):
        file.write(f"{_id},{_img_path.absolute()}\n")
        torch.save(_img_embedding, (tensors_path / f"{_id}.pt"))


    def update(self, model:Clip) -> Self:
        """
        Update an existing database: Delete nonexistent images and insert new ones.
        
        :param model: A model to embed new images.
        :type model: Clip
        :return: Return self to enable method chaining.
        :rtype: Self
        """        
        if self.img_root is None:
            raise Exception("The 'img_root' must exist. Open an existing database first that has a 'img_root'.")
        with catch_time(TIMING, "Database update"):
            _images = set(self._recurse_images(self.img_root))
            to_add = _images.difference(self.img_paths)
            to_remove = set(self.img_paths).difference(_images)
            add_ids = []
            for img_path in to_add:
                emb = model.embed_images([Image.open(img_path)])
                add_ids.append((self.add(img_path, emb), img_path, emb))
            remove_ids = []
            for img_path in to_remove:
                remove_ids.append(self.remove(img_path))
            
            with open(self.db_save_path / "db.csv", mode="r+") as f:
                lines = f.readlines()
                f.seek(0)

                for i, line in enumerate(lines):
                    if i == 0:
                        self._write_dbcsv_header(f)
                    else:
                        _id, _path = line.strip().split(",")
                        if int(_id) not in remove_ids:
                            f.write(line)
                        else:
                            (self.db_save_path / "tensors" / f"{_id}.pt").unlink()
                for id, img_path, emb in add_ids:
                    self._save_item(f, self.db_save_path / "tensors", id, img_path, emb)
                f.truncate()
        return self


    def embed_all(self, root_dir:str|Path, model:Clip) -> Self:
        """
        Find all images and embed them. Fill this database with the embeddings.
        
        :param root_dir: Root directory under which all images will be embedded.
        :type root_dir: str
        """
        if Path.is_file(self.db_save_path / "db.csv"):
            raise DatabaseExists(self.db_save_path)
        with catch_time(TIMING, "Database embed_all"):
            self.img_root = Path(root_dir).absolute()
            for img_path in self._recurse_images(Path(root_dir)):
                emb = model.embed_images([Image.open(img_path)])
                self.add(img_path, emb)
        return self
    


class DatabaseExists(Exception):
    def __init__(self, db_save_path) -> None:
        super().__init__(f"Database at '{db_save_path}' is already existing and would be overwritten.")