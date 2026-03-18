import os
os.environ['HIP_VISIBLE_DEVICES'] = "1" # wichtig damit integrated gpu nicht erkannt wird!!

from database import Database
from commandline import parse
from clip_model import Clip

from unittest import TestCase
import unittest
import sys
from pathlib import Path
from argparse import ArgumentError

import torch
from PIL import Image


class TestDatabase(TestCase):
    db_save_path = "a/b/c"
    images = "a/b/img"

    db:Database
    
    def setUp(self) -> None:
        super().setUp()
        self.db = Database(self.db_save_path, device="cpu")

    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_add(self):
        self.db.add("test_files/images/h9_20181001.jpg", torch.tensor([1]))
        self.db.add("test_files/images/h10_20181027.JPG", torch.tensor([2]))
        self.assertEqual(self.db.img_paths.index(Path("test_files/images/h10_20181027.JPG")), 1)
    
    def test_remove(self):
        self.db.add("test_files/images/h9_20181001.jpg", torch.tensor([1]))
        self.db.add("test_files/images/h10_20181027.JPG", torch.tensor([2]))
        self.assertEqual(self.db.img_paths.index(Path("test_files/images/h10_20181027.JPG")), 1)
        _id = self.db.ids[1]
        self.db.remove(Path("test_files/images/h10_20181027.JPG"))
        with self.assertRaises(ValueError):
            self.db.img_paths.index(Path("test_files/images/h10_20181027.JPG"))
        self.assertFalse(torch.tensor([2]) in self.db.img_embeddings)
        with self.assertRaises(ValueError):
            self.db.ids.index(_id)



class TestCommandline(TestCase):
    # search with existing:     main.py search -db /path/to/db "Ein Bild von ..."
    # create db:                main.py create -db /path/to/db --image-path /path/to/images
    # update db:                 main.py update -db /path/to/db
    
    def test_search(self):
        sys.argv = ["/blah/main.py", "search", "-db", "/path/to/db/db.csv", "Ein Bild von ..."]
        args = parse()
        self.assertEqual(args.command, "search")
        self.assertEqual(args.db_path, Path("/path/to/db/db.csv"))
        self.assertEqual(args.search_string,  "Ein Bild von ...")
        self.assertIsNone(args.image_path)
    
    def test_create(self):
        sys.argv = ["/blah/main.py", "create", "-db", "/path/to/db/db.csv", "--image-path", "/path/to/images"]
        args = parse()
        self.assertEqual(args.command, "create")
        self.assertEqual(args.db_path, Path("/path/to/db/db.csv"))
        self.assertEqual(args.image_path, Path("/path/to/images"))
        self.assertIsNone(args.search_string)

    def test_update(self):
        sys.argv = ["/blah/main.py", "update", "-db", "/path/to/db/db.csv"]
        args = parse()
        self.assertEqual(args.command, "update")
        self.assertEqual(args.db_path, Path("/path/to/db/db.csv"))
        self.assertIsNone(args.search_string)
        self.assertIsNone(args.image_path)

    def test_invalid(self):
        # update with image path is invalid
        sys.argv = ["/blah/main.py", "update", "-db", "/path/to/db/db.csv", "--image-path", "/path/to/images"]
        with self.assertRaises(SystemExit):
            args = parse()
        


class TestClip(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.clip = Clip()

    def test_image_embedding(self):
        img = Image.open("test_files/images/h9_20181001.jpg")
        embedding = self.clip.embed_images([img])
        self.assertEqual(embedding.shape, (1, 512))

    def test_text_embedding(self):
        embedding = self.clip.embed_text(["hallo", "tschüss"])
        self.assertEqual(embedding.shape, (2, 512))
    
    # def tearDown(self) -> None:
    #     super().tearDown()
    #     torch.cuda.synchronize()  # ROCm/HIP-Shutdown-Deadlock
        

if __name__ == "__main__":
    unittest.main()
