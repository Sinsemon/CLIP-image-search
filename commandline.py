from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

# search with existing:     main.py search -db /path/to/db "Ein Bild von ..."
# create db:                main.py create -db /path/to/db --image-path /path/to/images
# update db:                 main.py update -db /path/to/db

class Arguments:
    command:Literal["search", "create", "update"]
    db_path:Path
    search_string:str|None = None
    image_path:Path|None = None

    def __repr__(self) -> str:
        return "Arguments(" + ", ".join([f"{var}: {getattr(self, var)}" for var in vars(self)]) + ")"

def parse() -> Arguments:

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument("-db", "--db-path", type=Path, required=True)

    parser = ArgumentParser(
        prog="CLIP Image Search",
        description="Search for images using CLIP AI model."
    )
    subparsers = parser.add_subparsers(help="Commands:", required=True, dest="command")
    search_parser = subparsers.add_parser("search", help="Search in an existing database for images.", parents=[parent_parser])
    create_parser = subparsers.add_parser("create", help="Create a new database.", parents=[parent_parser])
    update_parser = subparsers.add_parser("update", help="Update an existing database. Insert new images into the database and remove nonexistent images.", parents=[parent_parser])

    search_parser.add_argument("search_string")
    create_parser.add_argument("--image-path", type=Path, help="Create a new database with the given directory as root. All images under this directory will be embedded (includes subdirectories).")

    return parser.parse_args(namespace=Arguments())
