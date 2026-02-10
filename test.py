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


# class TestDatabase(TestCase):
#     db_save_path = "test_files/db"
#     images = "test_files/images"

#     db:Database
    
#     def setUp(self) -> None:
#         super().setUp()
#         self.db = Database(self.db_save_path)

#     def tearDown(self) -> None:
#         return super().tearDown()
    
class TestCommandline(TestCase):
    # search with existing:     main.py search -db /path/to/db "Ein Bild von ..."
    # create db:                main.py manage -db /path/to/db --create /path/to/images
    # clean db:                 main.py manage -db /path/to/db --update
    
    def test_search(self):
        sys.argv = ["/blah/main.py", "search", "-db", "/path/to/db/db.csv", "Ein Bild von ..."]
        args = parse()
        self.assertEqual(args.db_path, Path("/path/to/db/db.csv"))
        self.assertEqual(args.search_string,  "Ein Bild von ...")
    
    def test_create(self):
        sys.argv = ["/blah/main.py", "manage", "-db", "/path/to/db/db.csv", "--create", "/path/to/images"]
        args = parse()
        self.assertEqual(args.db_path, Path("/path/to/db/db.csv"))
        self.assertEqual(args.create, Path("/path/to/images"))
        self.assertFalse(args.update)
        self.assertFalse(hasattr(args, "search_string"))

    def test_update(self):
        sys.argv = ["/blah/main.py", "manage", "-db", "/path/to/db/db.csv", "--update"]
        args = parse()
        self.assertEqual(args.db_path, Path("/path/to/db/db.csv"))
        self.assertEqual(args.update, True)
        self.assertFalse(hasattr(args, "search_string"))
        self.assertIsNone(args.create)

    def test_invalid(self):
        # update and create are not allowed together
        sys.argv = ["/blah/main.py", "manage", "-db", "/path/to/db/db.csv", "--update", "--create", "/path/to/images"]
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
