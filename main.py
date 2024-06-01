import argparse
import subprocess
from scripts.config import TEST_IMAGES_PATH

def run_face_detector(target_image):
    cmd = ['python', './scripts/face_detector.py', '-f', target_image]
    subprocess.run(cmd)

def run_demo(class_code):
    cmd = ['python', './scripts/compare_faces.py', '--class_code', class_code]
    subprocess.run(cmd)

def run_benchmark(num_classes):
    cmd = ['python', './run_benchmark/run_benchmark.py', "--num_classes" , str(num_classes)]
    subprocess.run(cmd)

def run_encoder(isNew, isBenchmark):
    cmd = ['python', './scripts/encoder.py']
    if isNew:
        cmd.append("--isNew")
    if isBenchmark:
        cmd.append("--isBenchmark")
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run project scripts.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--encode", action="store_true", help="Run the encode script.")
    group.add_argument("--demo", action="store_true", help="Run the demo script.")
    group.add_argument("--benchmark", action="store_true", help="Run the benchmark script.")

    # demo
    parser.add_argument("--class_code", type=str, help="The class code.")
    parser.add_argument("--no_detect", action="store_true", help="Skip face detection.")

    # encode
    parser.add_argument("--isNew", action="store_true", help="Encode new images.")
    parser.add_argument("--isBenchmark", action="store_true", help="Encode benchmark images.")

    # benchmark
    parser.add_argument("--num_classes", type=int, help="The number of classes to run the benchmark on.")
    parser.add_argument("--target_image", type=str, help="The target image to extract faces.")
    
    args = parser.parse_args()
    if args.demo:
        default_image = args.target_image if args.target_image else "8.jpg"
        class_code = args.class_code if args.class_code else "se1901"
        if not args.class_code:
            print(f"No --class_code provided. Using default class code: {class_code}")
        if not args.target_image:
            print(f"No --target_image provided. Using default image: {TEST_IMAGES_PATH}/{default_image}")
        if not args.no_detect:
            print("Skipping face detection.")
            run_face_detector(default_image)
        run_demo(class_code)
    elif args.benchmark:
        if args.num_classes:
            run_benchmark(args.num_classes)
        else:
            print("Please provide the number of classes using --num_classes 2")
    elif args.encode:
        if not args.isNew:
            print("No --isNew provided. Meaning no new images exist in stage.")
        if not args.isBenchmark:
            print("No --isBenchmark provided. Meaning script will run on database folder.")
        run_encoder(args.isNew, args.isBenchmark)
    