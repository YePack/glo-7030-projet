PATH_VIDEO='videos'
PATH_FRAMES='videos/frames'
VIDEO_NAME='game.mp4'
PATH_MODEL='test_phil_unet'
START_TIME=0005:11:000
END_TIME=00:06:00:000

all: validation train predict

install:
	brew install ffmpeg
	brew install imagemagick

submodule:
	git submodule update --init

validation:
	python3 -m src.semantic.create_data_training_setup

train:
	python3 -m src.semantic.training_script

predict:
	ffmpeg -i $(PATH_VIDEO)/$(VIDEO_NAME) -vf fps=30 $(PATH_FRAMES)/extracted_img%04d.png
	python3 -m src.data_creation.resize_images --path=$(PATH_FRAMES)
	python3 -m src.semantic.predict --path=$(PATH_MODEL) --folder=$(PATH_FRAMES)
	convert -delay 40 $(PATH_FRAMES)/prediction*.png $(PATH_VIDEO)/output.gif
	rm $(PATH_FRAMES)/*.png
