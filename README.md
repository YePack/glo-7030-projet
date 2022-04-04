# SemSeg Hockey Broadcasts

At first, this repo was created for our winter 2019 deep learning course at Laval University (GLO-7030) in Quebec City. Afterward, the project was continued and named **SemSeg Hockey Broadcasts**. The objective of this project is to train a model capable of learning the semantic of hockey broadcast images.

# Dataset

The images were taken from [NHL Game Recap](https://www.nhl.com/video/t-277753022) screenshots. The raw images (`.png`) and the corresponding labels (`.xml`) are stored in the `data/raw/` folder. The images are all resized and then labeled by a labeling tool called [cvat](https://github.com/opencv/cvat). See this [section](#resize-images) for the images resizing procedure and this [section](#launch-labeling-tool) for launching the labeling tool.

# Increase the dataset

Here is the procedure to follow in order to increase the dataset and label images.

## Prerequists

Docker must be installed. Here is the [installation link](https://runnable.com/docker/install-docker-on-macos) for macOS. It will be necessary for launching the labeling tool further.

After you installed Docker, you can install the Python libraries:

```
pip install -r requirements.txt
```

## Resize images
 
Once you have NHL recap screenshosts saved on your computer, you must first resize them before creating the labels. To do so, use the following script by specifying the path of the folder where your images are saved:

```
python -m src.data_creation.resize_images --path your_path
```

It should save all the `*.png` files inside that folder in the correct dimensions.

##  Launch labeling tool

Once you have resized images, you can launch the labeling tool and import those resized images to start a new labeling job.

To launch the tool for the first time:

1. Start Docker
2. Go in the directory : `lableling-tool/`
3. *Build* the docker image : `docker-compose build`
4. Run the container : `docker-compose up -d`
5. Create a superuser to have all the rights : `docker exec -it cvat bash -ic '/usr/bin/python3 ~/manage.py createsuperuser'`
6. Should now have access to the tool with [this link](http://localhost:8080/auth/register).

Otherwise, just start Docker and access the link.

## Generate labels

Once you launched the labeling tool, you are up and running to import you resized images inside a new *cvat job*. Here are the labels you should create in your job settings:

- Ice 
- Board
- Circlezone
- Circlemid
- Goal
- Blue
- Red
- Fo

## Split the XML labels

The last part before splitting the dataset and running the model is to extract the XML from the labeling tool and split it into separate XMLs (one for each file). To split the XML downloaded from the labeling tool:

```
python -m src.data_creation.parser.xml_splitter --file path_xml_file --dir dir_save_xmls
```

The very last step is to add the resized images and the accompagning XML to the `data/raw/` directory and push it to the repo.

# Generate predictions
>USE MAKE FILE! DESCRIPTION BELOW IS OUTDATED.

To generate predictions out of a sequence (mp4 video), you must have these two system libraries installed: `ffmpeg` and `ImageMagick`. To install them on Mac, use the following commands:

````
brew install ffmpeg
brew install imagemagick
````
Once the 2 libraries are installed, you can run the following shell script and ensure to have the right parameters at the beginning of the script.

```
./src/utils/video_predict.sh
```
