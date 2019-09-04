import random
import argparse
from nltk.corpus import wordnet as wn


def transitive_closure(synsets):

    hypernyms = set([])
    for s in synsets:
        paths = s.hypernym_paths()
        for path in paths:
            hypernyms.update((s,h) for h in path[1:] if h.pos() == 'n')
    return hypernyms


parser = argparse.ArgumentParser()
parser.add_argument('result_file')
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--sep', default='\t')
parser.add_argument('--target', default='mammal.n.01')
args = parser.parse_args()
print(args)


def main(args):

    target = wn.synset(args.target)
    print('target:', args.target)

    words = wn.words()

    nouns = set([])
    for word in words:
        nouns.update(wn.synsets(word, pos='n'))

    print( len(nouns), 'nouns')

    hypernyms = []
    for noun in nouns:
        paths = noun.hypernym_paths()
        for path in paths:
            try:
                pos = path.index(target)
                for i in range(pos, len(path)-1):
                    hypernyms.append((noun, path[i]))
            except Exception:
                continue

    hypernyms = list(set(hypernyms))
    print( len(hypernyms), 'hypernyms' )

    if not args.shuffle:
        random.shuffle(hypernyms)
    with open(args.result_file, 'w') as fout:
        for n1, n2 in hypernyms:
            print(n1.name(), n2.name(), sep=args.sep, file=fout)


if __name__ == '__main__':

    main(args)

