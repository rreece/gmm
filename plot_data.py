"""
generate_toy_data.py

author: Ryan Reece  <ryan.reece@cern.ch>
created: 2017-11-10
"""

import argparse
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


# options
#______________________________________________________________________________
def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles',  default=None, nargs='+',
            help='A positional argument.')
#    parser.add_argument('-o', '--out',  default='default.mdp', type=str,
#            help="Some toggle option.")
    return parser.parse_args()


#______________________________________________________________________________
def main():
    timestamp = time.strftime('%Y-%m-%d-%Hh%M')

    ops = options()

    for csvfile in ops.infiles:
        basename, ext = os.path.splitext( csvfile )
        plotfile = '%s.png' % basename

        print 'Reading %s' % csvfile
        data = read_csv(csvfile)
        print 'Plotting'
        plot_data(data, plotfile)
        print 'Saved plot to %s' % plotfile

    print 'Done'



#______________________________________________________________________________
def read_csv(csvfile):
    data = list()
    with open(csvfile, 'r') as f:
        reader = csv.reader(f, dialect='excel')
        for i_row, row in enumerate(reader):
            if i_row == 0:
                data.append( row )
            else:
                t_row = list()
                for c in row[:-1]:
                    t_row.append( float( c ) )
                s = row[-1]
                if s.isdigit():
                    t_row.append(int(s))
                else:
                    t_row.append(s)
                data.append(t_row)
        f.close()
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
    colors_map[4] = 'violet'
    colors_map[5] = 'cyan'
    colors_map[6] = 'khaki'
    colors_map[7] = 'teal'
    colors_map[8] = 'salmon'
        
    for row in data[1:]:
        _x = row[0]
        _y = row[1]
        _l = row[-1]
        x.append(_x)
        y.append(_y)
        colors.append( colors_map.get(_l, 'grey' ) )

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)

    plt.scatter(x, y, s=4, c=colors, marker='o')

    plt.xlabel(variables[0])
    plt.ylabel(variables[1])
#    plt.legend()

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    ax.grid(True)
    ax.set_axisbelow(True)

    plt.savefig(plotfile, bbox_inches='tight', dpi=100)
    plt.close()


#______________________________________________________________________________
def save_to_csv(data, csvfile='out.csv'):
    with open(csvfile, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)
        f.close()

#______________________________________________________________________________
if __name__ == '__main__': main()


