"""
Script for building a feature matrix given a directory of CSV or pickle files containing features. Can merge 20M rows x 10K columns across ~40 files given ~250GB of RAM in about an hour. Bigger data sets may require more time/RAM, while more files will require you to run this script on smaller subsets of the data before performing a final merge on the pre-built subset feature matrices.
@author: Rachael Cox <rachaelcox@utexas.edu>

Modified by Akshaya Rajaraman <arajar4@uic.edu>
- Updated make_fset() to accept a row with separate ID1 and ID2 
columns instead of a single space-delimited ID string, to match input .feat file format.
- Updated read_files() to drop 'ID1' and 'ID2' columns instead
  of 'ID', and use row-wise iteration for make_fset().
- Updated build_fmat() to accept optional suffixes parameter for
  handling duplicate column names (e.g. pearsonR) across feature files.
- Disabled fillna(0) in build_fmat() to preserve NaN values in
  the merged feature matrix.
- Added -s/--suffixes argument to CLI for labeling feature columns
  by file name when column names conflict.


"""

__author__ = "Rachael Cox (rachaelcox@utexas.edu)"
__version__ = "0.1.0"
__license__ = "MIT"

import argparse
from functools import reduce
import pickle
import pandas as pd
import os
import re
import time
import numpy as np
from datetime import datetime as dt

def make_fset(row):
    x1, x2 = row['ID1'], row['ID2']
    if x1 == x2:
        print(f"WARNING: Features for '{x1}-{x2}' (self-self PPI) detected ...")
        return None
    else:
        fset = frozenset({x1, x2})
        return(fset)

def read_files(data_dir, pickle_files=False):  
    fmat_list = []
    t0 = time.time()
    if not data_dir.endswith("/"):
        data_dir = data_dir+"/"
    if pickle_files:
        flist = [f for f in os.listdir(data_dir) if re.match('.*.pkl', f)]
        flist.sort()
        for f in flist:
            print(f' ► Reading {f} ...')
            with open(data_dir+f, 'rb') as handle:
                df = pickle.load(handle)
            df = df.round(4)
            #arajar4: ID1 and ID2 are two columns in my .feat document
            #df['frozen_pair'] = [make_fset(i) for i in df['ID']]
            df['frozen_pair'] = [make_fset(row) for _, row in df.iterrows()]
            #df.drop(['ID'], axis=1, inplace=True)
            df.drop(['ID1','ID2'], axis=1, inplace=True)
            df.dropna(subset=['frozen_pair'], inplace=True)
            fmat_list.append(df)
    else:
        flist = [f for f in os.listdir(data_dir) if not re.match('.*.pkl', f)]
        flist.sort()
        for f in flist:
            print(f' ► Reading {f} ...')
            df = pd.read_csv(data_dir+f)
            df = df.round(4)
            #df['frozen_pair'] = [make_fset(i) for i in df['ID']]
            df['frozen_pair'] = [make_fset(row) for _, row in df.iterrows()]
            #df.drop(['ID'], axis=1, inplace=True)
            df.drop(['ID1','ID2'], axis=1, inplace=True)
            df.dropna(subset=['frozen_pair'], inplace=True)
            fmat_list.append(df)
    return(fmat_list)

def get_left_join_idx(data_dir, pickle_files, left_file):
    if pickle_files:
        flist = [f for f in os.listdir(data_dir) if re.match('.*.pkl', f)]
        flist.sort()
        left_index = flist.index(left_file)
        return(left_index)
    else:
        flist = [f for f in os.listdir(data_dir)]
        left_index = flist.index(left_file)
        return(left_index)
    
def build_fmat(fmat_list, join_type='outer', left_index=None, suffixes=None):
    t0 = time.time()
    for df in fmat_list:
        df.set_index(['frozen_pair'], inplace=True)
    if join_type == 'outer':
        fmat = reduce(lambda x, y: x.join(y, how='outer', rsuffix=suffixes), fmat_list)
        #arajar4:I want to keep nan values here
        #fmat.fillna(0, inplace=True)
    elif join_type == 'left':
        fmat = fmat_list[left_index]
        fmat_list.pop(left_index)
        for df, suffix in zip(fmat_list, suffixes):
            fmat = fmat.join(df, how='left', rsuffix=suffixes)
            #arajar4:I want to keep nan values here
            #fmat.fillna(0, inplace=True)
    fmat.reset_index(inplace=True)
    fmat['ID'] = [' '.join(list(i)) for i in fmat['frozen_pair']]
    fmat.drop(['frozen_pair'], axis=1, inplace=True)
    feat_cols = [c for c in fmat.columns.values.tolist() if c != 'ID']
    fmat = fmat[['ID'] + feat_cols]
    print(f' ► Final feature matrix {fmat.shape}:')
    print(fmat.head())
    print(f' ► Total time to merge feature matrices: {time.time() - t0} seconds')
    return(fmat)

def write_fmat(fmat, data_dir, outfile_name):
    t0 = time.time()
    if not data_dir.endswith("/"):
        data_dir = data_dir+"/"
    if not outfile_name:
        outfile_name = data_dir+'featmat'
    print(f" ► Serializing joined matrix to {outfile_name+'.pkl'} ... ")
    fmat.to_pickle(outfile_name+'.pkl')
    print(f" ► Writing joined matrix to CSV {outfile_name} ...")
    fmat.to_csv(outfile_name, index=False)
    print(f' ► Total time to write out results: {time.time() - t0} seconds')


def main():
    
    t0 = time.time()
    # get files
    print(f'[{dt.now()}] Reading in feature files from {args.directory} ...')
    fmat_list = read_files(args.directory, args.pickle)
    # eval join type & build feature matrix
    print(f'[{dt.now()}] Merging feature matrices ...')
    if args.left_join_file:
        left_idx = get_left_join_idx(args.directory, args.pickle, args.left_join_file)
        print(args.suffixes)
        fmat = build_fmat(fmat_list, join_type='left', left_index=left_idx, suffixes=args.suffixes)
    else:
        print(args.suffixes)
        fmat = build_fmat(fmat_list, suffixes=args.suffixes)
    # write out results
    write_fmat(fmat, args.directory, args.outfile_name)
    print(f'[{dt.now()}] Done!')
    
    print(f"[{dt.now()}] ---------------------------------------------------------")
    print(f"[{dt.now()}] Total run time: {round((time.time()-t0)/60, 2)} minutes.")
    print(f"[{dt.now()}] ---------------------------------------------------------")
        
if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # specify directory (required)
    parser.add_argument("-d", "--directory", help="(Required) path to directory where feature files are located. Must be CSV or pickled files.")
    
    #specify labels for feature values if they have the same column name (eg. pearsonR)
    parser.add_argument("-s", "--suffixes", action="store", nargs='+', default=None, help="(optional) Specify the labels of the pearsonR values along with file name.")

    # specify if pickle files (default=false)
    parser.add_argument("-p", "--pickle", action="store_true", default=False, help="(Optional) specify this argument to indicate features are compressed into pickle files (will significantly speed up run time).")
    
    # specify if you want to left join on a file
    parser.add_argument("-l", "--left_join_file", action="store", default=None, help="(Optional) specify a file with IDs you want to use as a left join index (will significantly speed up run time).")

    # specify outfile name (default='featmat', written to the given data directory)
    parser.add_argument("-o", "--outfile_name", action="store", default=None, help="(Optional) specify the outfile path/name (defaults are 'featmat' and 'featmat.pkl', written to the given feature directory).")
    
    # specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    main()
