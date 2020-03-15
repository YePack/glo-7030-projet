import argparse
import os
from tqdm import tqdm

from src.player_detection.pipeline.capture_images import CaptureImages
from src.player_detection.pipeline.capture_image import CaptureImage
from src.player_detection.pipeline.predict import Predict
from src.player_detection.pipeline.annotate_image import AnnotateImage
from src.player_detection.pipeline.save_image import SaveImage
from src.player_detection.pipeline.utils import detectron


def parse_args():

    ap = argparse.ArgumentParser(
        description="Detectron2 image processing pipeline")
    ap.add_argument("-i",
                    "--input",
                    #required=True,
                    default="data/player_detection/test",
                    help="path to input image file or directory")
    ap.add_argument("-o",
                    "--output",
                    default="output",
                    help="path to output directory (default: output)")
    ap.add_argument("-p",
                    "--progress",
                    action="store_true",
                    help="display progress")
    ap.add_argument("-sb",
                    "--separate-background",
                    action="store_true",
                    help="separate background")

    # Detectron settings
    ap.add_argument(
        "--config-file",
        default="COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
        help=
        "path to config file (default: COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml)"
    )
    ap.add_argument("--config-opts",
                    default=[],
                    nargs=argparse.REMAINDER,
                    help="modify model config options using the command-line")
    ap.add_argument("--weights-file",
                    default="models/model_final.pth",
                    help="path to model weights file")
    ap.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="minimum score for instance predictions to be shown (default: 0.8)"
    )

    # Mutliprocessing settings
    ap.add_argument("--gpus",
                    type=int,
                    default=1,
                    help="number of GPUs (default: 1)")
    ap.add_argument("--cpus",
                    type=int,
                    default=0,
                    help="number of CPUs (default: 1)")

    return ap.parse_args()


def main(args):
    # Create output directory if needed
    os.makedirs(args.output, exist_ok=True)

    # Create pipeline steps
    capture_images = CaptureImages(args.input) \
        if os.path.isdir(args.input) else CaptureImage(args.input)

    cfg = detectron.setup_cfg(config_file=args.config_file,
                              weights_file=args.weights_file,
                              config_opts=args.config_opts,
                              confidence_threshold=args.confidence_threshold,
                              cpu=False if args.gpus > 0 else True)

    predict = Predict(cfg)

    annotate_image = AnnotateImage("vis_image", classes=["away", "home"])

    save_image = SaveImage("vis_image", args.output)

    # Create image processing pipeline
    pipeline = (capture_images | predict | annotate_image | save_image)

    # Iterate through pipeline
    try:
        for _ in tqdm(pipeline, disable=not args.progress):
            pass
    except StopIteration:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    args = parse_args()
    main(args)