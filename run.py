from main import main

# sys.argv = ["./main.py", "search", "--help"]
# sys.argv = ["./main.py", "create", "-db", "./test_files/db", "--image-path", "./test_files/images"]
# sys.argv = ["./main.py", "update", "-db", "./test_files/db"]
# sys.argv = ["./main.py", "search", "-db", "./test_files/db", "Ein Bild einer Blume."]

# sys.argv = ["./main.py", "create", "-db", "./test_files/db100", "--image-path", "./test_files/img100"]
args = ["search", "-db", "./test_files/db100", "Halloween Kürbisse"]

main(args)