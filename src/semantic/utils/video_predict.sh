
PATH_VIDEO='/home/vladyslav/Documents/test-video'
VIDEO_NAME='game.mp4'
PATH_MODEL='/home/vladyslav/Documents/R&D/glo-7030-projet/test_phil_unet'
NB_IMAGES_SEC=30
TIME_FROM=00:07:11.000
TIME_TO=00:07:18.000

ffmpeg -i $PATH_VIDEO/$VIDEO_NAME  -vf fps=$NB_IMAGES_SEC/1 $PATH_VIDEO/extracted_image%04d.png -hide_banner
python3 -m src.data_creation.resize_images --path $PATH_VIDEO

python3 -m src.semantic.predict --path $PATH_MODEL --folder $PATH_VIDEO

convert -delay 40 $PATH_VIDEO/prediction*.png $PATH_VIDEO/output.gif

rm $PATH_VIDEO/*.png