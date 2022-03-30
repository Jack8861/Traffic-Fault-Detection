# Traffic-Fault-Detection

This project uses the power of computer vision to detect when a vehicle breaks a certain traffic rule and scrapes its license plate to store it in the database.

## Description

- It can detect vehicles (car, auto, bike/2 wheelers).
- It can detect if a person is wearing a helmet of not.
- It can scrape the license plate from vehicles which break traffic rules.
- save the license plate numbers and related information in the database.

## Aim / Goal

- To implement computer vision ideas in the field of law inforcements.
- To automate law inforcement.
- To demonstrate implementation of data science in real life.

## Skillset

- Algorithmic knowledge (YOLOV5)
- Data collection (from online and offline videos, images)
- Data preparation (labeling images)
- Model training and testing
- Logic building
- Python Programming (backend)
- Web Development (flask)
- Cassandra Database (setup and usage)
- Version control (git, github)

## Dataset

- Online dataset:
    - https://www.kaggle.com/datasets/andrewmvd/helmet-detection
    - google images for helmet
- Offline:
    - videos of vehicles were captured by me near traffic signals and junctions
    - i manually picked images from the videos
    - labeled the images for training
    - images of vehicles (autos, 2 wheelers, cars)
    - helmet images
    - about 250 and 200 images where used to train the vehicle and helmet detection models respectively.
    - 100 extra images were used for testing the models.

## Demo

[YouTube Video link](https://youtu.be/cpAEbemEo2U)

## Tech Stack

<code><img height="80" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png"></code>
<code><img height="80" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/html/html.png"></code>
<code><img height="80" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/css/css.png"></code>
<code><img height="80" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/javascript/javascript.png"></code>
<code><img height="80" src="https://raw.githubusercontent.com/github/explore/bbd48b997e8d0bef63f676eca4da5e1f76487b56/topics/visual-studio-code/visual-studio-code.png"></code>
<code><img height="80" src="https://raw.githubusercontent.com/github/explore/8b79365c693905ff9adad384ab1534b5ab041cb9/topics/cassandra/cassandra.png"></code>
<code><img height="80" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/git/git.png"></code>
<code><img height="80" src="https://raw.githubusercontent.com/github/explore/d530d6a3a171a53f7b8eb4e9e005136e7ebd898f/topics/numpy/numpy.png"></code>
<code><img height="80" src="https://raw.githubusercontent.com/pandas-dev/pandas/761bceb77d44aa63b71dda43ca46e8fd4b9d7422/web/pandas/static/img/pandas.svg"></code>
<code><img height="80" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/opencv/opencv.png"></code>
<code><img height="80" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/ubuntu/ubuntu.png"></code>
<code><img height="80" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/flask/flask.png"></code>
- Labelimg
- easyocr
  
## Installation

Requirements

- Python 3.8
- YoloV5
- Cassandra 3.11
- Windows or ubuntu 20.04 (preferbly, setup Cassandra on dockers if working on windows)



## Setup and usage guide

- First clone the yolov5 repository in the folder you want to set it up and install the requirements

`
git clone https://github.com/ultralytics/yolov5  # clone
`

`
cd yolov5
`

- Then download this repository and have all these contents in the yolov5 folder
    - Take the vehicle.pt, helmet.pt, detect2.py and detect3.py files out of their respective

- Install the dependencies *using the requirements file provided in this repository*:

`
pip install -r requirements.txt  # install
`

### Information regarding the files provided:
- The vehicle.pt and helmet.pt are the trained weights which are used for detection
- The detect2.py is my own and basically it does the same as detect.py but does not create bounding boxes and instead returns those values
- The detect3.py is what is actually used by the application for detecting vehicles, helmets and scraping license plates
- The detect.py file which comes with yolov5 is used to detect and output images/videos. 

`
python detect.py --source <image/video location> --weights <vehicle.pt or helmet.pt>
`

### There are 2 traffic rules implemented:

- Does any vehicle break the traffic light (signal breaker)?


## How it works:

- First it needs a bounding box so that it can detect objects within it
    - The reason for this is that a traffic camera covers pedestrians and vehicles from other lanes so its best to create a bounding box.
    - And it also reduces unnessary load on the processor.
- Then it needs to know where the zebra crossing is
    - You can create a model for it. I collected the data but haven't labelled and trained the model for it.
- It needs the know the signal from the traffic signal.
    - This is currently set to red and for zebra crossing i have kept the its as 200 pixels from the bottom.
    - So if a vehicle tries to leave the frame from the bottom of the frame then it is considered as signal breaker
- OCR: easyocr is used in this project to detect the license plate and read it.
- Every time a 2 wheeler is detected it the helmet is checked.

## Documentation

- [High Level Documents](https://drive.google.com/file/d/1EAOnpQhf3ap8X5ZOHQGU3NjqJa8umhp7/view?usp=sharing)
- [Low Level Documents](https://drive.google.com/file/d/1oNYMINswUHTthdt66PbtEtuaT4rJnlwj/view?usp=sharing)
- [Wireframe](https://drive.google.com/file/d/1gLoHLRzVcfGwWxG5kmx0x4f_U5lHpy_8/view?usp=sharing)
- [Report](https://docs.google.com/presentation/d/14Pmn4SR93L7fRnGJvNQfJuKAf8PURd94/edit?usp=sharing&ouid=105403021575418724386&rtpof=true&sd=true)

 
## Challenges

- The first challenge i faced was understanding how yolov5 works because after some time of playing around with it i realized that i can't just use it as it is and had to make my own version of the detect.py for the project.wen
- Then the second challenge was the data collection and preparation as the online data sources can't be used as it is (The above give kaggle dataset has a lot of images with bicycle and construction hats and it does not represent the kind of images it will encounter in real life) and there was less data to build both a helmet detection and vehicle detection model (indian vehicles, autos, 2 wheelers)
    - I went out and recorded a few videos of vehicles on roads where there is dense traffic and also shown a few of those videos here for demo.
    - After collection i labelled the images using labelimg
- Then there where a few problems i faced in cassandra installation and setup but i quickely learnt how to deal with them after a lot of stackoverflow, datastax and github searches.
- I also had some issues with opencv and where basically i found out that some packages where not of the right versions needed and caused a issue with imread and other functions. (solution was to delete and recreate the virtual environment)

## What i Learnt

- YoloV5
- Usage of cassandra DB
- Usage of git, github.
- opencv
- logic building
- model building and testing

## Support

For support, email jackneutron786@gmail.com or contact me on linkedin.

  
## Usage
- You can use this project for further developing it and adding your work in it. If you use this project, kindly mention the original source of the project and mention the link of this repo in your report.
- You can add more traffic rule detection using new ideas.
  
## 🚀 About Me
- I am a data science enthusiast.
- Currently studying 4th year in CSE
- I have some helpful content on kaggle as well so, check it out if interested
  

## 🔗 Links
[![kaggle](https://img.shields.io/badge/Kaggle-000?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/jackfroster)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/syed-rahim-saqib-2505221b5/)

  
## License

[Apache License 2.0](https://github.com/Jack8861/Traffic-Fault-Detection/blob/main/LICENSE)

  
## Author

- [Syed Rahim Saqib](https://www.github.com/Jack8861)

  

