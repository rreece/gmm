"""
auto_gmm.py

author: Ryan Reece  <ryan.reece@cern.ch>
created: 2017-11-10
"""

#import argparse
import csv
import math
import os
import time

import numpy as np
np.random.seed(777) # for DEBUG

import matplotlib
matplotlib.use("Agg") # suppress the python rocketship icon popup
import matplotlib.pyplot as plt

plt.style.use('seaborn-deep')
plt.rcParams['legend.numpoints'] = 1

from sklearn.mixture import GaussianMixture

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

    print 'Hello. This is auto_data.py'

    name = 'hello'
    n_cycles = 5
    n_types = 0
    n_dims = 2
    purities = list()
    means = dict()
    covs = dict()

    for i_cycle in xrange(n_cycles):
        print 'Cycle %i' % i_cycle

        ## setup cluster parameters
        if i_cycle == 0:
            if n_dims == 2:
                n_types = 2
                purities = [0.6, 0.4]
                for i_type in range(n_types):
                    means[i_type] = [np.random.rand()*0.90+0.05, np.random.rand()*0.90+0.05]
                    covs[i_type] = [[np.random.rand()*0.005+0.001, np.random.rand()*0.001],[np.random.rand()*0.001, np.random.rand()*0.005+0.001]]
            elif n_dims == 3:
                n_types = 2
                purities = [0.6, 0.4]
                for i_type in range(n_types):
                    means[i_type] = [np.random.rand()*0.90+0.05, np.random.rand()*0.90+0.05, np.random.rand()*0.90+0.05]
                    covs[i_type] = [[np.random.rand()*0.005+0.001, np.random.rand()*0.001, np.random.rand()*0.001], [np.random.rand()*0.001, np.random.rand()*0.005+0.001, np.random.rand()*0.001], [np.random.rand()*0.001, np.random.rand()*0.001, np.random.rand()*0.005+0.001]]
                pass
            else:
                 assert False
        else:
            if n_dims == 2:
                n_new_types = int(np.random.rand()*1.0)+1 # rand = 1-2
                for _ in range(n_new_types):
                    purities.append( np.random.rand()*0.06+0.02 ) # rand = 0.02-0.08
                    means[n_types] = [np.random.rand()*0.90+0.05, np.random.rand()*0.90+0.05]
                    covs[n_types] = [[np.random.rand()*0.005+0.001, np.random.rand()*0.001],[np.random.rand()*0.001, np.random.rand()*0.005+0.001]]
                    n_types += 1
            elif n_dims == 3:
                n_new_types = int(np.random.rand()*1.0)+1 # rand = 1-2
                for _ in range(n_new_types):
                    purities.append( np.random.rand()*0.06+0.02 ) # rand = 0.02-0.08
                    means[n_types] = [np.random.rand()*0.90+0.05, np.random.rand()*0.90+0.05, np.random.rand()*0.90+0.05]
                    covs[n_types] = [[np.random.rand()*0.005+0.001, np.random.rand()*0.001, np.random.rand()*0.001], [np.random.rand()*0.001, np.random.rand()*0.005+0.001, np.random.rand()*0.001], [np.random.rand()*0.001, np.random.rand()*0.001, np.random.rand()*0.005+0.001]]
                    n_types += 1
            else:
                 assert False

        ## normalize purities
        _sum_purities = sum(purities)
        purities = [ p/_sum_purities for p in purities ]

        ## generate data for this cycle
        csvfile = 'd%i.csv' % i_cycle
        print 'Generating data n_types=%i, purities=%s' % (n_types, purities)
        print '    means=%s' % (means)
        print '    covs=%s' % (covs)
        data = generate_data(n_dims=n_dims, n_types=n_types, n_points=500, purities=purities, means=means, covs=covs)
        save_to_csv(data, csvfile)
        print 'Saved data to %s' % csvfile

        if i_cycle == 0:
            csvfile = 'p%i.csv' % i_cycle
            save_to_csv(data, csvfile)
            print 'Saved perfect labeled data to %s' % csvfile
            continue

        csvfile_prev = 'p%i.csv' % (i_cycle-1)
        csvfile = 'd%i.csv' % i_cycle

        inf = InferenceGMM()
        inf.initialize(name)

        ## load past processed data
        print 'Loading %s' % csvfile_prev
        inf.load_csv(csvfile_prev, labeled=True)

        ## load new data batch
        print 'Loading %s' % csvfile
        inf.load_csv(csvfile, labeled=False)

        ## inference
        print 'Inferring missing labels'
        results = inf.infer_labels(q_threshold=10.0, q_threshold_local=20.0, n_neighbors=5, plotfile='gmm_contours_%i.png' % i_cycle, interupt=False)

        ## plot results
        print 'Plotting results'
        plot_results(results, name='gmm_results_%i' % i_cycle)

        ## save
        csvfile = 'p%i.csv' % i_cycle
        inf.write_csv(csvfile)
        print 'Saved processed data to %s' % csvfile
        inf.close()

    print 'Done'



#______________________________________________________________________________
def generate_data(n_dims=2, n_types=2, n_points=400, purities=None, means=None, covs=None, labeled=True):

    assert 1 <= n_dims <= 10
    assert 1 <= n_types <= 10

    if purities == None:
        assert n_types > 0
        purity_share = 1.0/n_types
        purities = tuple([purity_share]*n_types)
    assert abs(sum(purities) - 1.0) < 1e-6

    assert n_types == len(purities) == len(purities) == len(means) == len(covs)

    var_names = dict()
    label_names = dict()
    var_names[0] = 'x'
    var_names[1] = 'y'
    var_names[2] = 'z'
    var_names[3] = 'a'
    var_names[4] = 'b'
    var_names[5] = 'c'
    var_names[6] = 'd'
    var_names[7] = 'e'
    var_names[8] = 'f'
    var_names[9] = 'g'
    for i_type in range(10):
        label_names[i_type] = i_type

    data_header = list()
    for i_dim in range(n_dims):
        data_header.append( var_names[i_dim] )
    data_header.append('label')

    data_gen = list()
    for i_type in range(n_types):
        data_for_type = list()
        _p = purities[i_type]
        _n = int(round(n_points*_p))
        label_name = None
        if labeled:
            label_name = label_names[i_type]
        for rep in np.random.multivariate_normal(means[i_type], covs[i_type], _n):
            rep_list = list(rep)
            rep_list.append(label_name)
            data_for_type.append(rep_list)
        data_gen.extend(data_for_type)

    np.random.shuffle(data_gen)

    data = list()
    data.append(data_header)
    data.extend(data_gen)
    return data


#______________________________________________________________________________
def save_to_csv(data, csvfile='out.csv'):
    with open(csvfile, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)
        f.close()


#==============================================================================
class InferenceGMM(object):

    #__________________________________________________________________________
    def __init__(self):
        self.name = None
        self.header = None
        self.data = list()
        self.labels = set()
        self.means = list()
        self.covariances = list()
        self.precisions = list()
        self.counts = dict()
        self.n_components = 0
        self.gmm = None

    #__________________________________________________________________________
    def initialize(self, name):
        self.name = name
        self.gmm = None

    #__________________________________________________________________________
    def load_csv(self, csvfile, labeled):
        _data = self.read_csv(csvfile, labeled)
        if _data:
            self.data.extend(_data)

    #______________________________________________________________________________
    def read_csv(self, csvfile, labeled):
        """
        TODO: generalize column types
        """
        data = list()
        with open(csvfile, 'r') as f:
            reader = csv.reader(f, dialect='excel') # TODO: do we need excel?
            for i_row, row in enumerate(reader):
                if i_row == 0:
                    ## NOTE: assuming there is always a header
                    self.header = row
                else:
                    t_row = list()
                    for c in row[:-1]:
                        t_row.append( float( c ) )
                    if labeled:
                        s = row[-1]
                    else:
                        s = ''
                    if s.isdigit():
                        t_row.append(int(s))
                    elif s == '':
                        t_row.append(None)
                    else:
                        t_row.append(s)
                    data.append(t_row)
            f.close()
        return data

    #__________________________________________________________________________
    def write_csv(self, csvfile='out.csv'):
        with open(csvfile, 'w') as f:
            writer = csv.writer(f)
            writer.writerows([self.header])
            writer.writerows(self.data)
            f.close()

    #__________________________________________________________________________
    def infer_labels(self, q_threshold=5.99, q_threshold_local=5.99,
            n_neighbors=3, plotfile='gmm_contours.png', update=True, interupt=False):
        """
        hyperparams: conf_threshold or q_threshold

        returns list: [(rep, label, conf, Q), ...]
        """
        n_dims = None
        self.counts = dict()
        _X = list()
        _labels = list()
        _X_unlab = list()
        _means_dict = dict()
        for row in self.data:
            _label = row.pop()
            _rep = row
            if n_dims is None:
                n_dims = len(_rep)
            if _label is None:
                _X_unlab.append( _rep )
            else:
                _X.append( _rep )
                _labels.append( _label )
                self.labels.add( _label )
                self.counts.setdefault(_label, 0)
                self.counts[_label] += 1
                _means_dict.setdefault(_label, [0.]*n_dims)
                for _i, _x in enumerate(_rep):
                    _means_dict[_label][_i] += _x

        _labels_list = self.counts.keys()
        _labels_list.sort()
        _weights_init = list()
        _means_init = list()
        _total_count = len(_X)
        for _l in _labels_list:
            for _i in xrange(n_dims):
                _means_dict[_l][_i] /= self.counts[_l]
            _m = [ _means_dict[_l][_i] for _i in xrange(n_dims) ]
            _means_init.append( np.array( _m ) )
            _weights_init.append( float(self.counts[_l])/_total_count )

        self.n_components = len(self.counts)
        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type='full',
                                weights_init=_weights_init,
                                means_init=_means_init,
                                warm_start=False)

        self.gmm.fit(_X, _labels)
        self.weights     = self.gmm.weights_
        self.means       = self.gmm.means_
        self.covariances = self.gmm.covariances_
        self.precisions  = self.gmm.precisions_
        
        ## DEBUG
        print 'Results of initial fit:'
        print 'weights_', self.gmm.weights_
        print 'means_', self.gmm.means_
        print 'covariances_', self.gmm.covariances_
        print 'precisions_', self.gmm.precisions_
        self.plot_contours(_X, _labels, plotfile=plotfile)

        results = list()
        ## loop over unlabeled data
        for _rep in _X_unlab:

            ## check unlabeled data one at a time
            X_unlab = np.array( [ _rep ] )

            predicted_labels = self.gmm.predict(X_unlab)
            predicted_probs = self.gmm.predict_proba(X_unlab)
            Qs = -2* self.gmm.score_samples(X_unlab)

            assert len(X_unlab) == len(predicted_labels) == len(predicted_probs) == 1

            ## not a real loop, one at a time
            for i_unlab, rep in enumerate(X_unlab):
                Q = Qs[i_unlab]
                print 'rep: %s, Q: %.6g' % (rep, Q)

                ## get closest cluster
                closest_mean_rep, closest_label, closest_dist = self.get_closest_cluster(rep)
                print 'Closest mean rep:', closest_mean_rep, closest_label, closest_dist

                ## use closest cluster for label
                label = closest_label

#                label = predicted_labels[i_unlab]
                conf = predicted_probs[i_unlab][label]

                ## check for global anomaly
                if Q > q_threshold:
                    print 'Global anomaly: ', rep, Q, label, conf

                    _X_local = list()
                    for row in self.data:
                        _label = row.pop()
                        _rep = row
                        if _label == closest_label:
                            _X_local.append( _rep )

                    assert _X_local, closest_mean_rep

#                   avg_precision = sum([ self.gmm.precisions_[i][j][j] for j in range(len(self.gmm.precisions_)) ])/len(self.gmm.precisions_)
                    avg_precision = np.identity(n_dims)*100 # TODO

                    ## if closest cluster has more than the minimum points (n_neighbors)
                    ## check for local anomaly
                    if len(_X_local) >= n_neighbors:


                        ## fit single-component GMM
                        local_gmm = GaussianMixture(n_components=1, covariance_type='full',
                                weights_init=np.array([1.0]),
                                means_init=[np.array(closest_mean_rep)],
                                precisions_init=[avg_precision])
                        local_gmm.fit(_X_local)
                        local_Qs = -2* local_gmm.score_samples(X_unlab)
                        local_Q = local_Qs[i_unlab]

                        if local_Q > q_threshold_local:
                            print 'Local anomaly: ', rep, local_Q, closest_label, conf

                            ## get new label
                            max_label = max(_labels)
                            label = max_label + 1
                            conf = 0.0
                            self.n_components += 1
                            assert self.n_components == label + 1

                            _weights = np.append(np.array(self.gmm.weights_), [1.0/len(self.data)], axis=0)
                            _weights = _weights/sum(_weights)
                            _means = np.append(np.array(self.gmm.means_), [rep], axis=0)
                            _precisions = self.gmm.precisions_
                            _precisions = np.append(_precisions, [avg_precision], axis=0)
                            print 'prefit _means', _means
                            print 'prefit _precisions', _precisions
                            self.gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', warm_start=False,
                                    weights_init=_weights, means_init=_means, precisions_init=_precisions)

                        else:
                            print 'Not local anomaly: ', rep, local_Q, closest_label, conf
                            print local_gmm.means_
                            print local_gmm.precisions_
                            Q = local_Q
                            label = closest_label

                    else:
                        ## Low-count scenario. Check if new point is close enough to
                        ## group by assumption.
                            
                        ## TODO: Need a better test here for the case of nearby
                        ## anomalies that will get incorrectly labeled the same.
                        ## Something like a k-NN vote instead of just the closest?

                        grouping_dist = 0.2
                        if closest_dist > grouping_dist:
                            print 'Local anomaly, low count: ', rep, closest_label

                            ## get new label
                            max_label = max(_labels)
                            label = max_label + 1
                            conf = 0.0
                            self.n_components += 1
                            assert self.n_components == label + 1

                            _weights = np.append(np.array(self.gmm.weights_), [1.0/len(self.data)], axis=0)
                            _weights = _weights/sum(_weights)
                            _means = np.append(np.array(self.gmm.means_), [rep], axis=0)
                            _precisions = self.gmm.precisions_
                            _precisions = np.append(_precisions, [avg_precision], axis=0)
                            print 'prefit _means', _means
                            print 'prefit _precisions', _precisions
                            self.gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', warm_start=False,
                                    weights_init=_weights, means_init=_means, precisions_init=_precisions)
                        else:
                            print 'Not local anomaly, low count: ', rep, Q, closest_label, conf
                            label = closest_label

                else:
                    ## not a global anomaly
#                    label = predicted_labels[i_unlab]
                    label = closest_label
                    conf  = predicted_probs[i_unlab][label]
                    print 'Infer:  %s'  % label

                results.append( (rep, label, conf, Q) )

                ## update _X, _labels, and metadata
                if update:
                    _X.append( X_unlab[i_unlab] )
                    _labels.append( label )
                    self.labels.add( label )
                    self.counts.setdefault(label, 0)
                    self.counts[label] += 1
                    self.gmm.fit(_X, _labels)
                    self.weights     = self.gmm.weights_
                    self.means       = self.gmm.means_
                    self.covariances = self.gmm.covariances_
                    self.precisions  = self.gmm.precisions_
                    ## DEBUG
                    print 'Results of new fit:'
                    print 'n_components', self.n_components
                    print 'weights_', self.gmm.weights_
                    print 'means_', self.gmm.means_
                    print 'covariances_', self.gmm.covariances_
                    print 'precisions_', self.gmm.precisions_
                    print 'counts', self.counts

                ## update total data store
                ## TODO: this is not efficient
                if update:
                    self.data = list()
                    self.labels = set()
                    for i_rep, rep in enumerate(_X):
                        _label = _labels[i_rep]
                        row = list(rep)
                        row.append(_label)
                        self.data.append( row )
                        self.labels.add(_label)

            if interupt:
                _ = raw_input('Press enter to continue') ## DEBUG

        return results

    #__________________________________________________________________________
    def get_closest_cluster(self, rep):
        min_count_for_using_mahalanobis = 5
        closest_mean_rep = None
        closest_label = None
        min_d = 9e12
#        min_euclid_d = 9e12
#        min_euclid_l = None
#        min_mahala_d = 9e12
#        min_mahala_l = None
#        min_euclid_d_high = 9e12
#        min_euclid_l_high = None
#        min_mahala_d_high = 9e12
#        min_mahala_l_high = None
        sorted_labels = list(self.labels)
        sorted_labels.sort()
        msg = '%s  %s' % (self.means, sorted_labels)
        assert len(self.means) == len(sorted_labels) == len(self.precisions), msg

#        for _r, _l, _p in zip(self.means, sorted_labels, self.precisions):
#
#            _de = self.calc_dist(rep, _r) ## Euclidean
#            _dm = self.calc_dist(rep, _r, _p) ## Mahalanobis
#            _c = self.counts[_l]
#            if _de < min_euclid_d:
#                min_euclid_d = _de
#                min_euclid_l = _l
#            if _dm < min_mahala_d:
#                min_mahala_d = _dm
#                min_mahala_l = _l
#
#            if _c >= min_count_for_using_mahalanobis:
#                if _de < min_euclid_d:
#                    min_euclid_d_high = _de
#                    min_euclid_l_high = _l
#                if _dm < min_mahala_d:
#                    min_mahala_d_high = _dm
#                    min_mahala_l_high = _l
#
#        ## set closest_label
#        if self.counts[min_mahala_l] >= min_count_for_using_mahalanobis:
#            closest_label = min_mahala_l
#        elif min_mahala_l == min_euclid_l:
#            closest_label = min_mahala_l
#        else:
#            closest_label = min_mahala_l_high
#
#        ## set closest_mean_rep
#        for _r, _l in zip(self.means, sorted_labels):
#            if _l == closest_label:
#                closest_mean_rep = _r
#
#        ## set min_d (always Euclidean)
#        min_d = self.calc_dist(rep, closest_mean_rep) ## Euclidean

        for _r, _l, _p in zip(self.means, sorted_labels, self.precisions):
#            _d = self.calc_dist(rep, _r, _p) ## Mahalanobis
            _d = self.calc_dist(rep, _r) ## Euclidean

            if _d < min_d:
                min_d = _d
                closest_mean_rep = _r
                closest_label = _l

        print 'get_closest_cluster', msg, closest_mean_rep, closest_label
        return closest_mean_rep, closest_label, min_d
    
    #__________________________________________________________________________
    def calc_dist(self, r1, r2, precision=None):
        """
        If precision is None, calculate Euclidean distance.
        Otherwise, calculate the Mahalanobis distance
        assuming precision is the inverse covariance matrix.
        """
        dist2 = 0.0
        if precision is None:
            for _x1, _x2 in zip(r1, r2):
                _dx = _x1 - _x2
                dist2 += _dx*_dx
        else:
            ##  calculate (x1 - x2).T * VI * (x1 - x2)
            Z = r1 - r2
            tmp = np.dot(precision, Z)
            dist2 = np.dot(Z.T, tmp)
        return math.sqrt(dist2)

    #__________________________________________________________________________
    def plot_contours(self, X, labels, plotfile='gmm_contours.png'):
        grid = np.linspace(0.0, 1.0)
        gX, gY = np.meshgrid(grid, grid)
        h = len(gX.ravel())
        vs = [gX.ravel(), gY.ravel()]
        n_dims = len(X[0])
        for _  in xrange(2, n_dims):
            vs.append([0.1]*h)
        XX = np.vstack(vs).T
        Qs = -2*self.gmm.score_samples(XX)
        Qs = Qs.reshape(gX.shape)

        fig, ax = plt.subplots()
        fig.set_size_inches(6, 5)

        contour_plot = plt.contour(gX, gY, Qs, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=20.0),
                    levels=np.linspace(0, 20, 5))
        colorbar_plot = plt.colorbar(contour_plot, shrink=0.8, extend='both', label='Q = -2 log(L)')

        color_map = {
            0 : 'blue',
            1 : 'red',
            2 : 'green',
            3 : 'orange',
            4 : 'violet',
            5 : 'cyan',
            6 : 'khaki',
            7 : 'teal',
            8 : 'salmon',
            }

        variables = {
            0 : 'x',
            1 : 'y',
            2 : 'z',
            3 : 'a',
            4 : 'b',
            5 : 'c',
            6 : 'd',
            7 : 'e',
            8 : 'f',
            9 : 'g',
            }

        xs = [p[0] for p in X]
        ys = [p[1] for p in X]
        colors = [ color_map.get(_l, 'grey' ) for _l in labels ]

        plt.scatter(xs, ys, s=4, c=colors, marker='o')

        plt.xlabel(variables[0])
        plt.ylabel(variables[1])

        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        ax.grid(True)
        ax.set_axisbelow(True)
        plt.savefig(plotfile, bbox_inches='tight', dpi=100)
        plt.close()

    #__________________________________________________________________________
    def close(self):
        pass

#______________________________________________________________________________
def plot_results(results, name='results'):
    _Qs = list()
    _confs = list()
    for rep, label, conf, Q in results:
        _Qs.append(Q)
        _confs.append(conf)

    ## plot Q
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    n, bins, patches = plt.hist(np.array(_Qs), 50, range=(0.0, 50.0), facecolor='green', alpha=0.75)
    plt.xlabel('Q = -2 log(L)')
    ax.grid(True)
    ax.set_axisbelow(True)
    plotfile = '%s_Q.png' % name
    plt.savefig(plotfile, bbox_inches='tight')
    plt.close()

    ## plot confs
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    n, bins, patches = plt.hist(np.array(_confs), 50, range=(0.0, 1.0), facecolor='green', alpha=0.75)
    plt.xlabel('Predictive probability')
    ax.grid(True)
    ax.set_axisbelow(True)
    plotfile = '%s_conf.png' % name
    plt.savefig(plotfile, bbox_inches='tight')
    plt.close()


#______________________________________________________________________________
if __name__ == '__main__': main()


