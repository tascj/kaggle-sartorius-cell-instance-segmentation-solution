import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location='cpu')
    torch.save(dict(state_dict=ckpt['model']), args.output)


if __name__ == '__main__':
    main()
