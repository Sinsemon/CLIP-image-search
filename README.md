# CLIP Image Seach

## MVP

- [ ] Initial command line interface
  - [x] Create DB with image embeddings at location x or default location
    - [x] Find all images under given path
    - [x] Embed all images with progress bar
    - [x] Save embeddings and corresponding image path in file
    - [ ] Progress bar
    - [ ] default location
  - [x] Update database to check for deleted / new images
    - [x] Open existing database
    - [x] Check for new / deleted images
    - [x] Remove deleted images from DB
    - [x] Add new images to DB
  - [x] Search the n most relevant images for a given string in given db
    - [x] Open existing DB
    - [x] Embed text
    - [x] Compare to DB and find the $n$ most similar images
  - [x] Improve CLI
- [ ] Documentation and nice Readme

## Later

- [ ] Create nice UI
