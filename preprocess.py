import json
import jsonlines
from argparse import ArgumentParser, Namespace
import csv

def main(args):
    dataset = {}
    train_file = jsonlines.open(args.train)
    valid_file = jsonlines.open(args.valid)
    file = open("train.json", "w", encoding='utf-8')
    for line in train_file.iter():
        json.dump(line, file, ensure_ascii=False)
    file = open("valid.json", "w", encoding='utf-8')
    #f = open("predict.json", "w", encoding='utf-8')
    for line in valid_file.iter():
        json.dump(line, file, ensure_ascii=False)
#        D = line
#        D["title"] = ""
#        json.dump(D, f, ensure_ascii=False)
        

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--train", type=str, default="data/train.jsonl")
    parser.add_argument("--valid", type=str, default="data/public.jsonl")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)


