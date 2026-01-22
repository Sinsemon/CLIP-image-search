# CLIP Image Seach

## MVP

- [ ] Initial command line interface
  - [ ] Create DB with image embeddings at location x or default location
    - [ ] Find all images under given path
    - [ ] Embed all images with progress bar
    - [ ] Save embeddings and corresponding image path in file
  - [ ] Update database to check for deleted / new images
    - [ ] Open existing database
    - [ ] Check for new / deleted images
    - [ ] Remove deleted images from DB
    - [ ] Add new images to DB
  - [ ] Search the n most relevant images for a given string in given db
    - [ ] Open existing DB
    - [ ] Embed text
    - [ ] Compare to DB and find the $n$ most similar images
- [ ] Documentation and nice Readme

## Later

- [ ] Create nice UI
