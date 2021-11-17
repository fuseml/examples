import argparse
import json
import numpy as np
import os.path
import re
import sys

from typing import AnyStr, Tuple

def create_request_data(img):
    signature = "serving_default"
    data_obj = {"signature_name": signature, 'instances': img.tolist()}
    return data_obj

def load_or_generate_data(numpy_file: AnyStr, generate_shape: Tuple, batch_size: int, transpose: bool):
    if numpy_file:
        if not os.path.isfile(numpy_file):
            print(f"Could not find numpy dataset file at location '{os.path.abspath(numpy_file)}'")
            exit(1)
        imgs = np.load(numpy_file, mmap_mode='r', allow_pickle=False)
        if transpose:
            imgs = imgs.transpose((0, 3, 1, 2))
        imgs = imgs - np.min(imgs)  # Normalization 0-255
        imgs = imgs / np.ptp(imgs) * 255  # Normalization 0-255
        # imgs = imgs[:,:,:,::-1] # RGB to BGR
        # imgs = imgs.astype(np.uint8)
        if batch_size:
            while batch_size > imgs.shape[0]:
                imgs = np.append(imgs, imgs, axis=0)

        img = imgs[:batch_size]
    elif generate_shape is not None:
        img = np.random.rand(batch_size or 1, *generate_shape)
    else:
        print("Either a numpy input file or a shape used to generate data must be supplied.")
        exit(1)

    data = create_request_data(img)
    return data


def parse_arguments():

    def shape_arg_type(value: AnyStr) -> Tuple[int, int, int]:
        """Parse a shape argument in the form [w,h,c] or [c,h,w].

        Args:
            value (AnyStr): string representation, as read from command line

        Raises:
            argparse.ArgumentTypeError: if argument isn't formatted correctly

        Returns:
            Tuple[int, int, int]: returns the parsed shape
        """
        regex=re.compile(r"^\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*$")
        match=regex.match(value)
        if not match:
            raise argparse.ArgumentTypeError(f"Invalid '[c,h,w]' or '[w,h,c]' format for --generate_shape value: {value}")
        return [int(match.group(1)), int(match.group(2)), int(match.group(3))]

    parser = argparse.ArgumentParser(description='Convert an image dataset in numpy form into a JSON TensorFlow '
                                                 'serving API format or generate a JSON dataset from scratch to be '
                                                 'used as a request body for TensorFlow serving REST API calls')   
    parser.add_argument("--images_numpy_path", type=str, help="input numpy file with data in shape [n,w,h,c] or [n,c,h,w]")
    parser.add_argument("--generate_shape", type=shape_arg_type, default=None, help="shape of dataset to generate in [w,h,c] or [c,h,w] form")
    parser.add_argument('--transpose_input', action="store_true", help='do a NHWC to NCHW input transposing')
    parser.add_argument("--batch_size", type=int, default=None, help="batch size (defaults to input dataset size)")
    parser.add_argument("--output", type=str, required=True, help="path to output JSON file")
    return parser.parse_args(sys.argv[1:])


def main():
    args = parse_arguments()
    print(' Command line options:')
    print('--images_numpy_path     : ',args.images_numpy_path or "N/A")
    print('--generate_shape        : ',args.generate_shape or "N/A")
    print('--transpose_input       : ',args.transpose_input)
    print('--batch_size            : ',args.batch_size or "auto")
    print('--output                : ',args.output)
    
    data = load_or_generate_data(args.images_numpy_path, args.generate_shape, args.batch_size, args.transpose_input)
    try:
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"I/O error while saving JSON to output file "
                f"'{os.path.abspath(args.output)}': {e.strerror}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error while saving JSON to output file "
                f"'{os.path.abspath(args.output)}': {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
