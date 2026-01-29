from database import Database

from unittest import TestCase
import unittest

import torch


class TestDatabase(TestCase):
    db_save_path = "test_files/db"
    images = "test_files/images"

    db:Database
    
    def setUp(self) -> None:
        super().setUp()
        self.db = Database(self.db_save_path)

    def tearDown(self) -> None:
        return super().tearDown()
    
    


if __name__ == "__main__":
    unittest.main()
