# CLIP Image Search

Have you ever wondered where that one beautiful image you took from the Louvre in Paris has gone?
Do you want to find all the images of your beautiful roses in the garden?
This is the solution: semantically browse your images using `openai/clip-vit-base-patch16`!

Warning: This program is still very early in development and has very restricted functionality.

## Installation

### For AMD GPUs on Windows using conda

Check this list of compatible AMD GPUs: [System requirements for Windows and compatibility](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html).
I installed PyTorch on my AMD GPU following [this documentation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html).

First, clone this repository and `cd` into it.

Create a virtual environment:

```
conda create -n torch python=3.12
conda activate torch
```

Install PyTorch from the AMD repo:

```
pip install --no-cache-dir ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz

pip install --no-cache-dir ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchaudio-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchvision-0.24.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl
```

Install the remaining dependencies:

```
pip install -r requirements.txt
```

### Other systems and GPUs / CPUs (not tested)

First, clone this repository and `cd` into it.

Then create a virtual environment and install the dependencies:

```
conda create -n torch python=3.12
conda activate torch
pip install torch transformers
```

## Usage

Activate the virtual environment and cd into this repository's root directory.

First you need to create a "database" where all the embeddings are stored.
`db/path` is the storage location of the database, and `image/path` is the root directory containing your images.
All images in this directory and in all subdirectories will be embedded.
This step may take a while depending on the number of images.

```
python main.py create -db <db/path> --image-path <image/path>
```

Now you can search your images.

```
python main.py search -db <db/path> "A beautiful image of the Louvre in Paris"
```

If you want to sync your database and the image directory, use the following command.
This will add new images to the database and remove deleted ones.

```
python main.py update -db <db/path>
```

## Still to do

- [ ] Initial CLI
  - [x] Create DB with image embeddings at location x or default location
    - [x] Find all images under the given path
    - [x] Embed all images
    - [x] Save embeddings and corresponding image paths in file
    - [ ] Progress bar
    - [ ] default location
  - [x] Update database to check for deleted / new images
    - [x] Open existing database
    - [x] Check for new / deleted images
    - [x] Remove deleted images from DB
    - [x] Add new images to DB
  - [x] Search the $n$ most relevant images for a given string in given DB
    - [x] Open existing DB
    - [x] Embed text
    - [x] Compare to DB and find the $n$ most similar images
- [ ] Documentation and Readme

### Later

- [ ] Create nice GUI
