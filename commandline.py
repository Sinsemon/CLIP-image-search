from argparse import ArgumentParser
from pathlib import Path

# search with existing:     main.py search -db /path/to/db "Ein Bild von ..."
# create db:                main.py manage -db /path/to/db --create /path/to/images
# clean db:                 main.py manage -db /path/to/db --update

class Arguments:
    db_path:Path
    search_string:str
    create:Path
    update:bool

    def __repr__(self) -> str:
        return "Arguments(" + ", ".join([f"{var}: {getattr(self, var)}" for var in vars(self)]) + ")"

def parse() -> Arguments:

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument("-db", "--db-path", type=Path, required=True)

    parser = ArgumentParser(
        prog="CLIP Image Search",
        description="Search for images using CLIP AI model."
    )
    subparsers = parser.add_subparsers()
    search_parser = subparsers.add_parser("search", help="Search in an existing database for images.", parents=[parent_parser])
    manage_parser = subparsers.add_parser("manage", help="Create or update an existing database.", parents=[parent_parser])

    search_parser.add_argument("search_string")
    mng_group = manage_parser.add_mutually_exclusive_group(required=True)
    mng_group.add_argument("--create", type=Path, help="Create a new database with the given image folder as root. All images under the folder will be embedded.")
    mng_group.add_argument("--update", action="store_true", help="Insert new images into the database and remove nonexistent images.")

    return parser.parse_args(namespace=Arguments())
