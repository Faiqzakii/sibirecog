import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sign to text: Command that parse a video stream and recognizes signs')
    parser.add_argument("-v", "--video", type=str, nargs='?')
    parser.add_argument("-t", '--train', action="store_true")
    parser.add_argument("-n", "--no-evaluate", action="store_true")
    args = parser.parse_args()

    print(args)