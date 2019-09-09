import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def main(embedding_file, output_file):

    targets = ['mammal.n.01', 'beagle.n.01', 'canine.n.02', 'german_shepherd.n.01',
               'collie.n.01', 'border_collie.n.01',
               'carnivore.n.01', 'tiger.n.02', 'tiger_cat.n.01', 'domestic_cat.n.01',
               'squirrel.n.01', 'finback.n.01', 'rodent.n.01', 'elk.n.01',
               'homo_sapiens.n.01', 'orangutan.n.01', 'bison.n.01', 'antelope.n.01',
               'even-toed_ungulate.n.01', 'ungulate.n.01', 'elephant.n.01', 'rhinoceros.n.01',
               'odd-toed_ungulate.n.01', 'mustang.n.01', 'liger.n.01', 'lion.n.01', 'cat.n.01', 'dog.n.01']

    targets = list(set(targets))
    print(len(targets), ' targets found')

    # load embeddings
    print("read embedding_file:", embedding_file)
    embeddings = pd.read_csv(embedding_file, header=None, sep="\t", index_col=0)
    print("plot")
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.cla()  # clear things for fresh plot

    for n in targets:
        z = embeddings.loc[n]
        if isinstance(z, pd.DataFrame):
            continue  # if the index is non-unique
        print(z.at[1], z.at[2])
        x, y = z.at[1], z.at[2]
        print(z, x, y)
        if n == 'mammal.n.01':
            ax.plot(x, y, 'o', color='g')
            ax.text(x + 0.01, y + 0.01, n, color='r', alpha=0.6)
        else:
            ax.plot(x, y, 'o', color='y')
            ax.text(x + 0.01, y + 0.01, n, color='b', alpha=0.6)
    if args.output_file:
        plt.savefig(args.output_file)
    else:
        plt.show()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('embedding_file')
    parser.add_argument('-o', '--output_file', default=None)
    args = parser.parse_args()

    main(args.embedding_file, args.output_file)
