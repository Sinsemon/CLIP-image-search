from database import Database
from commandline import parse

from unittest import TestCase
import unittest
import sys
from pathlib import Path

import torch


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

    def test_update(self):
        sys.argv = ["/blah/main.py", "manage", "-db", "/path/to/db/db.csv", "--update"]
        args = parse()
        self.assertEqual(args.db_path, Path("/path/to/db/db.csv"))
        self.assertEqual(args.update, True)



if __name__ == "__main__":
    unittest.main()
