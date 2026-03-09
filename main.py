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


def main(arguments:list[str]|None = None):
    args_parsed = parse(arguments)
    database = Database(args_parsed.db_path)
    model = Clip()

    if args_parsed.command == "search":  # search
        # search
        text_embedding = model.embed_text([args_parsed.search_string])
        database.load()
        print(database.get_similar(text_embedding, n=5))

    elif args_parsed.command == "create":  # create DB
        database.embed_all(args_parsed.image_path, model).save()

    else:  # update DB
        database.load().update(model)

if __name__ == "__main__":
    main()
