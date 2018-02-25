import csv
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def parse(s):
    s = s.strip()
    if s == 'none':
        return None
    try:
        return float(s)
    except ValueError:
        return s


color_map = {'SDB': 0, 'RR': 1, 'RM': 2, 'Vanilla': 3, 'Other': 4}
all_colors = [plt.cm.Accent(i) for i in np.linspace(0, 0.9, len(color_map))]


def alg_to_color(s):
    alg_type = s.split('-')[-1]
    if alg_type in color_map:
        color_ind = color_map[alg_type]
    else:
        color_ind = color_map['Other']
    return all_colors[color_ind]


def sizes(rrbs):
    smallest = min([rrb for rrb in rrbs if rrb is not None])
    sizes = [0 for _ in range(len(rrbs))]
    for i in range(len(sizes)):
        rrb = rrbs[i]
        if rrb is not None:
            sizes[i] = 500 * rrb
        else:
            sizes[i] = 500 * smallest
    return sizes


def make_scatter_plot(data_set, file_name, save_name):
    with open(file_name, 'r') as f:
        table_reader = csv.reader(f, delimiter=', ')
        labels = [parse(s) for s in next(table_reader)[1:]]
        errors = [parse(s) for s in next(table_reader)[1:]]
        biases = [parse(s) for s in next(table_reader)[1:]]
        rrbs = [parse(s) for s in next(table_reader)[1:]]

    # print(labels, errors, biases, rrbs)
    colors = [alg_to_color(s) for s in labels]
    plt.scatter(biases, errors, c=colors, s=sizes(rrbs))
    plt.xlabel('Average unsigned bias')
    plt.ylabel('Average error rate')

    x1, x2, y1, y2 = plt.axis()
    plt.axis((0, .35, y1, y2))
    # print(labels)
    dataset_customization(data_set, labels, biases, errors)

    plt.savefig(save_name, bbox_inches='tight')
    plt.show()


def dataset_customization(data_set, labels, biases, errors):
    alg_names = [s.split('-')[0] for s in labels]
    patches = [mpatches.Patch(color=all_colors[color_map[alg_type]], label=alg_type) for alg_type in color_map.keys()]
    offsets = [(11, 0) for _ in range(len(alg_names))]
    if data_set == 'german':
        offsets = [(11, -4), (11, -4), (11, -10), (11, 0), (8, -4), (8, -4), (11, -3), (11, 0),
                   (0, -19), (11, 0), (12, 0), (11, 0), (11, 0), (11, 0)]
        plt.legend(patches, [alg_type for alg_type in color_map.keys()])
    if data_set == 'singles':
        offsets[11] = (9, -8)
        plt.legend(patches, [alg_type for alg_type in color_map.keys()], loc=7)
    if data_set == 'adult':
        offsets = [(-33, 0), (9, 0), (9, 0), (11, 0), (10, 0), (11, 0), (-1, 10), (-5, -20),
                   (10, -10), (11, 0), (11, 0), (11, 0), (9, 1), (11, 0), (11, 0)]
        plt.legend(patches, [alg_type for alg_type in color_map.keys()])

    for label, x, y, offset in zip(alg_names, biases, errors, offsets):
        plt.annotate(label, xy=(x, y), xytext=offset, textcoords='offset points')
        # , xytext = (5, 5))
        # textcoords = 'offset points', ha = 'right', va = 'bottom',
        # bbox = dict(boxstyle = 'round, pad=0.5', fc = 'yellow', alpha = 0.5),
        # arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3, rad=0'))


if __name__ == '__main__':
    # make_scatter_plot('german', 'tables/german_avgs_only.csv', 'plots/german_scatter_plot.pdf')
    # make_scatter_plot('singles', 'tables/singles_avgs_only.csv', 'plots/singles_scatter_plot.pdf')
    make_scatter_plot('adult', 'tables/adult_avgs_only.csv', 'plots/adult_scatter_plot.pdf')
