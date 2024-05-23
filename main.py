import argparse
import subprocess
from config import TEST_IMAGES_PATH

def run_face_detector(target_image):
    cmd = ['python', 'face_detector.py', '-f', target_image]
    subprocess.run(cmd)

def run_demo(class_code):
    cmd = ['python', 'compare_faces.py', '--class_code', class_code]
    subprocess.run(cmd)

def run_benchmark(num_classes):
    cmd = ['python', 'run_benchmark.py', "--num_classes" , str(num_classes)]
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Run project scripts.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--demo", action="store_true", help="Run the demo script.")
    group.add_argument("--benchmark", action="store_true", help="Run the benchmark script.")

    parser.add_argument("--class_code", type=str, help="The class code.")
    parser.add_argument("--num_classes", type=int, help="The number of classes to run the benchmark on.")
    parser.add_argument("--target_image", type=str, help="The target image to extract faces.")
    args = parser.parse_args()
    if args.demo:
        if args.class_code:
            default_image = args.target_image if args.target_image else "8.jpg"
            if not args.target_image:
                print(f"No --target_image provided. Using default image: {TEST_IMAGES_PATH}/{default_image}")
            run_face_detector(default_image)
            run_demo(args.class_code)
        else:
            print("Please provide a class code using --class_code 'SE1900'")
    elif args.benchmark:
        if args.num_classes:
            run_benchmark(args.num_classes)
        else:
            print("Please provide the number of classes using --num_classes 2")


if __name__ == "__main__":
    main()