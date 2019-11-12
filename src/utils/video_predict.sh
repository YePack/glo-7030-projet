
PATH_VIDEO='/Users/stephanecaron/Downloads/test-video/montreal-vegas.mp4'
NB_IMAGES_SEC=2
TIME_FROM=00:07:11.000
TIME_TO=00:07:18.000

ffmpeg -i $PATH_VIDEO -ss $TIME_FROM -to $TIME_TO -vf fps=$NB_IMAGES_SEC/1 /Users/stephanecaron/Downloads/test-video/resized_prediction%04d.png -hide_banner

python3 -m src.predict --path '/Users/stephanecaron/Downloads/semseg-test/unet_dice' --folder '/Users/stephanecaron/Downloads/test-video/'

convert -delay 40 /Users/stephanecaron/Downloads/test-video/prediction*.png /Users/stephanecaron/Downloads/test-video/output.gif

rm /Users/stephanecaron/Downloads/test-video/*.png