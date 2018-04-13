import gzip
import argparse
import logging
import json
import pickle as pkl
import spacy
import itertools

from tqdm import tqdm
from random import randint,shuffle
from collections import Counter


def count_lines(file):
    count = 0
    for _ in file:
        count += 1
    file.seek(0)
    return count


def data_generator(data,msg="Reviews"):
    try:
        with gzip.open(args.input,"r") as f:
            for x in tqdm(f,desc=msg,total=count_lines(f)):
                yield json.loads(x)
    except Exception:
        with open(args.input,"r") as f:
            for x in tqdm(f,desc=msg,total=count_lines(f)):
                yield json.loads(x)


def to_array_comp(doc):
        return [[w.orth_ for w in s] for s in doc.sents]


def custom_pipeline(nlp):
    return (nlp.tagger, nlp.parser,to_array_comp)


def build_dataset(args,kv):

    print("Building dataset from : {}".format(args.input))
    print("-> Building {} random splits".format(args.nb_splits))

    nlp = spacy.load('en', create_pipeline=custom_pipeline)
    d0 = [(z[kv["user_id"]],z[kv["item_id"]],z[kv["rating"]]) for z in tqdm(data_generator(args.input,"getting data"),desc="reading file")]
    data = [(z[0],z[1],tok,z[2]) for z,tok in zip(d0,nlp.pipe((x[kv["review"]] for x in data_generator(args.input,"tokenizing")), batch_size=args.bsize, n_threads=8))]

    shuffle(data)

    splits = [randint(0,args.nb_splits-1) for _ in range(0,len(data))]
    count = Counter(splits)

    print("Split distribution is the following:")
    print(count)

    return {"data":data,"splits":splits,"rows":("user_id","item_id","review","rating")}


def main(args):

    if args.yelp:
        keys_values = {"user_id":"user_id","item_id":"business_id","review":"text","rating":"stars"}
    else:
        keys_values = {"user_id":"reviewerID","item_id":"asin","review":"reviewText","rating":"overall"}

    ds = build_dataset(args,keys_values)
    pkl.dump(ds,open(args.output,"wb"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str, default="sentences.pkl")
    parser.add_argument("--nb_splits",type=int, default=5)
    parser.add_argument("--bsize",type=int, default=100000)
    parser.add_argument("--yelp",action="store_true")

    args = parser.parse_args()

    main(args)
