import os
os.environ['HIP_VISIBLE_DEVICES'] = "1" # wichtig damit integrated gpu nicht erkannt wird!!
from time import time_ns
import_start = time_ns()

from commandline import parse
from database import Database
from clip_model import Clip
from const import TIMING

from pathlib import Path
import sys

from PIL import Image

if TIMING:
    print("imported: ", (time_ns() - import_start) / 1e9)

# sys.argv = ["./main.py", "manage", "-db", "./test_files/db", "--create", "./test_files/images"]
sys.argv = ["./main.py", "manage", "-db", "./test_files/db", "--update"]
# sys.argv = ["./main.py", "search", "-db", "./test_files/db", "Ein Bild einer Blume."]




args = parse()
database = Database(args.db_path)
model = Clip()

if hasattr(args, "search_string"):  # search
    # search
    text_embedding = model.embed_text([args.search_string])
    database.load()
    print(database.get_similar(text_embedding, n=5))

elif not args.update:  # create DB
    database.embed_all(args.create, model).save()

else:  # clean DB
    database.load().update(model)


