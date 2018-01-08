"""
generate_toy_data.py

author: Ryan Reece  <ryan.reece@cern.ch>
created: 2017-11-10
"""

#import argparse
import time
import csv
import os

import numpy as np
np.random.seed(777) # for DEBUG

import matplotlib
matplotlib.use("Agg") # suppress the python rocketship icon popup
import matplotlib.pyplot as plt

plt.style.use('seaborn-deep')
plt.rcParams['legend.numpoints'] = 1


## options
##______________________________________________________________________________
#def options():
#    parser = argparse.ArgumentParser()
#    parser.add_argument('infiles',  default=None, nargs='+',
#            help='A positional argument.')
##    parser.add_argument('-o', '--out',  default='default.mdp', type=str,
##            help="Some toggle option.")
#    return parser.parse_args()


#______________________________________________________________________________
def main():
    timestamp = time.strftime('%Y-%m-%d-%Hh%M')

#    ops = options

    print 'Hello. This is generate_toy_data.py'

    csvfile = 'toy2d_0.csv'

    print 'Generating data'
    data = generate_data(n_dim=2, n_types=2, n_points=500, purities=(0.6, 0.4))
    print 'Saving csv'
    save_to_csv(data, csvfile)
    print 'Saved data to %s' % csvfile

    csvfile = 'toy2d_1.csv'

    print 'Generating data'
    data = generate_data(n_dim=2, n_types=3, n_points=500, purities=(0.6, 0.38, 0.02), unlabeled=True)
    print 'Saving csv'
    save_to_csv(data, csvfile)
    print 'Saved data to %s' % csvfile

    csvfile = 'toy2d_2.csv'

    print 'Generating data'
    data = generate_data(n_dim=2, n_types=4, n_points=500, purities=(0.57, 0.37, 0.03, 0.03), unlabeled=True)
    print 'Saving csv'
    save_to_csv(data, csvfile)
    print 'Saved data to %s' % csvfile

    print 'Done'



#______________________________________________________________________________
def generate_data(n_dim=2, n_types=2, n_points=400, purities=None, unlabeled=False, n_anomalies=0):

    assert n_dim == 2
    assert n_types in (2, 3, 4)

    if purities == None:
        assert n_types > 0
        purity_share = 1.0/n_types
        purities = tuple([purity_share]*n_types)
    assert sum(purities) == 1.0

    ## HACK: hardcoded
    var_names = dict()
    label_names = dict()
    means = dict()
    covs = dict()
    if n_dim == 2:
        var_names[0] = 'x'
        var_names[1] = 'y'
        label_names[0] = 0
        label_names[1] = 1
        label_names[2] = 2
        label_names[3] = 3
        means[0] = [0.0, 0.0]
        means[1] = [1.0, 1.0]
        means[2] = [2.0, -1.0]
        means[3] = [0.8, -1.5]
        covs[0]  = [[0.1, 0.0], [0.0, 0.1]]
        covs[1]  = [[0.3, 0.0], [0.0, 0.1]]
        covs[2]  = [[0.02, 0.0], [0.0, 0.02]]
        covs[3]  = [[0.02, 0.0], [0.0, 0.02]]
    elif n_dim == 3:
        pass

    data_header = list()
    for i_dim in range(n_dim):
        data_header.append( var_names[i_dim] )
    data_header.append('label')

    data_gen = list()
    for i_type in range(n_types):
        data_for_type = list()
        _p = purities[i_type]
        _n = int(round(n_points*_p))
        label_name = None
        if not unlabeled:
            label_name = label_names[i_type]
        for rep in np.random.multivariate_normal(means[i_type], covs[i_type], _n):
            rep_list = list(rep)
            rep_list.append(label_name)
            data_for_type.append(rep_list)
        data_gen.extend(data_for_type)

    ## HACK: hardcoded
    data_anom = None
    if n_dim == 2:
        if n_anomalies > 0:
            data_anom     = [
#                [ 0.0,  0.0,  None], 
#                [ 0.1,  0.1,  None], 
#                [ 1.0,  1.0,  None], 
#                [ 0.0,  1.0,  None], 
#                [ 1.0,  0.0,  None], 
#                [ 2.0, -1.5,  None], 
                [ 2.0, -1.0,  None],
                [ 2.2, -1.1,  None],
                            ]
    elif n_dim == 3:
        pass

    if data_anom:
        data_gen.extend(data_anom)

    np.random.shuffle(data_gen)

    data = list()
    data.append(data_header)
    data.extend(data_gen)
    return data


#______________________________________________________________________________
def plot_data(data, plotfile='scatter.png'):
    """
    header = data[0]
    variables = header[:-1]

    ## make data_dict for DataFrame
    data_dict = dict()
    for var in variables:
        data_dict[var] = list()
    for row in data[1:]:
        for i_col, col in enumerate(row):
            var = header[i_col]
            data_dict[var].append(col)

    ## make  plot
    df = pd.DataFrame(data_dict)
    ax = df.plot(x=variables[0], y=variables[1:], marker='o', markersize=8)
    fig = ax.get_figure()
    plt.margins(x=0.1, y=0.1, tight=True)
    if fig:
        fig.savefig(plotfile, bbox_inches='tight')
        plt.close()
    """

    header = data[0]
    variables = header[:-1]

    x = list()
    y = list()
    colors = list()

    colors_map = dict()
    colors_map[0] = 'blue'
    colors_map[1] = 'red'
    colors_map[2] = 'green'
    colors_map[3] = 'orange'
        
    for row in data[1:]:
        _x = row[0]
        _y = row[1]
        _l = row[2]
        x.append(_x)
        y.append(_y)
        colors.append( colors_map.get(_l, 'grey' ) )

    plt.scatter(x, y, s=4, c=colors, marker='o')

    plt.xlabel(variables[0])
    plt.ylabel(variables[1])
#    plt.legend()

    plt.savefig(plotfile, bbox_inches='tight')
    plt.close()


#______________________________________________________________________________
def save_to_csv(data, csvfile='out.csv'):
    with open(csvfile, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)
        f.close()

#______________________________________________________________________________
if __name__ == '__main__': main()


