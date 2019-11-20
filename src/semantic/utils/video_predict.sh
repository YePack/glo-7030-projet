
PATH_VIDEO='/Users/stephanecaron/Downloads/test-video'
VIDEO_NAME='montreal-vegas.mp4'
PATH_MODEL='/Users/stephanecaron/Downloads/unet'
NB_IMAGES_SEC=2
TIME_FROM=00:07:11.000
TIME_TO=00:07:18.000

ffmpeg -i $PATH_VIDEO/$VIDEO_NAME -ss $TIME_FROM -to $TIME_TO -vf fps=$NB_IMAGES_SEC/1 $PATH_VIDEO/extracted_image%04d.png -hide_banner

python3 -m src.semantic.utils.resize_images --path $PATH_VIDEO

python3 -m src.semantic.predict --path $PATH_MODEL --folder $PATH_VIDEO

convert -delay 40 $PATH_VIDEO/prediction*.png $PATH_VIDEO/output.gif

rm $PATH_VIDEO/*.png