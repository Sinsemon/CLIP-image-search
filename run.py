from main import main

# args = ["search", "--help"]
# args = ["create", "-db", "./test_files/db", "--image-path", "./test_files/images"]
# args = ["update", "-db", "./test_files/db"]
# args = ["search", "-db", "./test_files/db", "The Moon"]

args = ["create", "-db", "./test_files/db100", "--image-path", "./test_files/img100"]
# args = ["search", "-db", "./test_files/db100", "Halloween Kürbisse"]

# args = ["create", "-db", "./test_files/db2025", "--image-path", "D:\Bilder\Persönliche Bilder\2025"]
# args = ["search", "-db", "./test_files/db2025", "Zwei Leute, die mit Neoprenanzug und Helm in einem Bach sitzen"]

main(args)