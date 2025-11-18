import os
import argparse
import glob

from PIL import Image


def make_gif(input_dir, duration=500):
    """
    Make a GIF from PNG images in the specified directory

    Parameters
    ----------
    input_dir : containing PNG images
    duration  : duration of each frame in milliseconds (default: 500 ms)
    """
    image_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))

    images = []
    for filename in image_files:
        images.append(Image.open(filename))

    images[0].save(os.path.join(input_dir, 'out.gif'), save_all=True, append_images=images[1:], duration=duration)
    print(f'GIF saved to {os.path.join(input_dir, "out.gif")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a GIF from a directory of PNG images.')
    parser.add_argument('input_dir', type=str, help='Input directory containing PNG images')
    parser.add_argument('--duration', type=int, default=500, help='Duration of each frame in milliseconds')
    args = parser.parse_args()

    make_gif(args.input_dir, duration=args.duration)
