


"""
Utility functions and classes for our capstone project
Protein Pathfinders
"""

import os, os.path
from collections import Counter
from random import choice
from datetime import datetime
import pickle
from random import randint, random
from math import log2, log, ceil

import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score
from warnings import warn, filterwarnings
#filterwarnings('ignore', category=UserWarning)



def smape_score(y_true, y_pred, factor=None, add_one=True):
    """
    y_true, y_pred must be numpy arrays (1d or 2d)
    
    Based on the SMAPE formula:
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    
    factor: 0.5 for [0%, 200%]   1.0 for [0%, 100%]
    epsilon: arbitrary small value to avoid division by zero
    """
    
    # modification as per the updated conditions in the challenge
    add_one = add_one or 0
    y_true = np.array(y_true) + int(add_one)
    y_pred = np.array(y_pred) + int(add_one)
    
    assert y_true.shape == y_pred.shape, "both arrays must be of equal shape"
    
    factor = float(factor or 0.5)
    
    if factor not in (1.0, 0.5):
        raise ValueError("factor must be either 1 or 0.5")
    
    epsilon = 1e-05  # arbitrary small value to avoid division by zero
    numerator = np.abs(y_pred - y_true)
    denominator = np.maximum((np.abs(y_true) + np.abs(y_pred)) * factor, epsilon)
    assert numerator.shape == denominator.shape, "must be of equal shapes"
    return np.sum(numerator / denominator) * (100 / y_true.size)
    


def rmse_score(y_true, y_pred):
    """
    Wrapper function around mean_squared_error
    """
    assert y_true.shape == y_pred.shape, "both arrays must be of equal shape"
    return mean_squared_error(y_true, y_pred) ** (1/2)



# not needed anymore
def drop_columns(df, threshold = 0.05, copy=True):
    """
    drops columns with too many nans
    keeps only the columns with the proportion of nans less than 'threshold'
    """
    cols = (df.isnull().sum(axis=0) / len(df)).sort_values() < threshold
    cols = cols[cols==True].index
    
    if copy:
        df = df.copy()
    return df[cols]



# not needed anymore
def log_transform(df, copy=True):
    """
    log-transforms element-wise but doesn't touch the first column (visit_month)
    """
    if copy:
        df = df.copy()
    df.iloc[:, 1:] = np.log(df.iloc[:, 1:]).replace({np.inf: np.nan, -np.inf:np.nan})
    return df



def get_sample_weights(categorical_array):
    """
    Returns sample weights based on the provided categories
    """
    counter = Counter(categorical_array)
    k = len(counter)
    return [1.0 / counter[v] / k for v in categorical_array]



def get_data(drop_columns=None,
             multiindex=False,
             all_nans_targets=['drop', 'separate'],
             return_patient_ids=False,
             full=False):
    """
    TODO: update the docs
    Returns a single dataframe.
    Loads, joins and sets the index as per arguments
    
    Parameters
    ----------
    full = True : include the rows which don't contain protein/peptide measurenments
    
    multiindex = True  : the index of the dataframe is in the format (55, 0)
    multiindex = False : the index of the dataframe is in the format '55_0'
                         'patien_id' gets dropped, 'visit_month' becomes the first column
    
    drop_columns = [...] : drops the columns named in the list
    drop_columns = "proteins" : drops all protein columns
    drop_columns = "peptides" : drops all peptide columns
        note: format   drop_columns=['proteins', 'visit_month']   works as well 
    
    drop_rows
        not implementd. Note: if elimination of rows should take place
            it must take place in this function, and not anywhere later in the process
    
    return_patient_ids = True
        aslo return patient_ids (to be used to compute sample_weights)
    """
    
    # redefine the drop_columns argument to make the mechanics of dropping cols flexible
    drop_columns = [drop_columns] if type(drop_columns) is str else drop_columns or []
    if not (hasattr(drop_columns, '__len__') and all(type(e) is str for e in drop_columns)):
        raise TypeError("drop_columns must be string(s)")
    
    # file names
    file_names = ("train_clinical_data.csv", 
                  "train_peptides.csv", 
                  "train_proteins.csv")
    
    dfs = None
    
    for path in [("..", "data"), ("data",), ("",)]:
        if not all(os.path.exists(os.path.join(*(path + (file_name,)))) 
                   for file_name in file_names):
            continue
        
        dfs = (pd.read_csv(os.path.join(*(path + (file_name,)))) for file_name in file_names)
        
        if dfs is not None:
            break
    
    # check if the dataframes were erad
    if dfs is None:
        raise RuntimeError("Failed to read the files. Try again.")
    
    # after the 3 df's were successfully read...
    df1, df2, df3 = dfs

    # pivot
    targets = df1.set_index(['patient_id', 'visit_month']).drop('visit_id', axis=1)
    pivot_peptides = pd.pivot(data=df2, index=['patient_id', 'visit_month'], columns='Peptide', values='PeptideAbundance')
    pivot_protines = pd.pivot(data=df3, index=['patient_id', 'visit_month'], columns='UniProt', values='NPX')
    
    # choose to drop proteins or peptides otherwise join both dataframes
    if "proteins" in drop_columns and "peptides" in drop_columns:
        df = pd.DataFrame(index=targets.index)
        warn("The obseravtions which contain missing values in all of the four targets will be dropped", UserWarning)
    elif "proteins" in drop_columns:
        df = pivot_peptides
    elif "peptides" in drop_columns:
        df = pivot_protines
    else:
        df = pivot_peptides.merge(pivot_protines, left_index=True, right_index=True, how='left')
    
    # full or only the rows with peptide/protein counts ?
    if full:
        df = df.merge(targets, how='outer', left_index=True, right_index=True)
    else:
        df = df.merge(targets, left_index=True, right_index=True, how='left')
    
    # reorder the columns so that the 4 target variables are at the very end
    if True: # set to False to deactivate this behavior if not needed
        col = "upd23b_clinical_state_on_medication"
        df = df.reindex(columns=list(df.columns)[:-5] + [col] + list(df.columns)[-5:-1], copy=False)
      
    # get the patient_id for sample weights
    patient_ids = df.index.get_level_values(0).values
    
    # (55, 0) or '55_0' format for the index?
    if not multiindex:
        nx = map(lambda t: f"{t[0]}_{t[1]}", df.index.values)
        df.reset_index(inplace=True)
        df.index = nx

    # drop columns specified by the user
    if drop_columns:
        if not set(drop_columns).issubset(df.columns.tolist() + ["proteins", "peptides", "patient_id", "visit_month"]):
            raise ValueError("bad column name(s) in 'drop_columns'")
        df.drop(list(drop_columns), axis=1, inplace=True, errors='ignore')

    # what to do with the rows that with all 4 targets equal to nan?
    new_data = None
    target_names = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]
    mask = (df[target_names].isnull().sum(1) == len(target_names)).values
    
    if all_nans_targets == 'separate':
        new_data = df.loc[mask].drop(target_names, axis=1)
    
    if all_nans_targets in ('separate', 'drop'):
        df.dropna(axis=0, how='all', subset=target_names, inplace=True)
        patient_ids = patient_ids[np.bitwise_not(mask)]
        
    elif type(all_nans_targets) not in (list, tuple, type(None)):
        raise ValueError(f"all_nans_targets argument not understood: {all_nans_targets}")

    assert patient_ids is None or len(patient_ids) == len(df), "err"
    
    patient_ids = patient_ids if return_patient_ids else None
    
    # return 1,2 or 3 objects
    return df if all(e is None for e in (new_data, patient_ids))\
           else [e for e in (df, new_data, patient_ids) if e is not None]



def preprocess(data, 
               upd23b_clinical_state_on_medication=False,
               target=False,
               target_names=None,
               copy=True):
    """
    Preprocess the data according to agreed procedures.
    Imputes missing values in the target.
    Expects a dataframe with the targets within.
    
    Parameters
    ----------
    data
        the whole DataFrame containing the targets
    
    upd23b_clinical_state_on_medication=True
        replace the values in "upd23b_clinical_state_on_medication" according to MAPPING (see below)
    
    target_names
        identify the names of the targets in the dataframe
        the default is ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]
    
    copy=False
        manipulates on the dataframe by reference, not by copy
    """
    # view or copy ?
    df = data if not copy else data.copy()
    
    # preprocess the data
    col = "upd23b_clinical_state_on_medication"
    if upd23b_clinical_state_on_medication and col in df.columns:
        MAPPING = {'Off': 0, 'On': 1, np.nan: 0}   # discussed this
        df[col] = df[col].replace(MAPPING).astype(int)

    # preprocess the targets
    target_names = target_names or ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

    #define methods
    f = {'mean': SimpleImputer(strategy='mean').fit_transform,
         'median': SimpleImputer(strategy='median').fit_transform,
         'zeros': lambda df: df.fillna(value=0.0),
         'zero': lambda df: df.fillna(value=0.0)}

    if target is None or target is False:
        pass # do nothing

    elif target is True:
        raise ValueError("True is ambiguous for target preprocessing. Be specific.")
    
    elif str(target) in f.keys():
        df[target_names] = f[target](df[target_names])

    elif type(target) in (list, tuple):
        if len(target) != len(target_names):
            raise ValueError("the given methods list length is not equal to the number of targets.")
        
        for col,method in zip(target_names, target):
            df[col] = f[method](df[col].to_frame())
            
    else:
        raise ValueError("target argument is not understood")
    
    return df    
    
    

def isolate_target(dataframe, return_arrays=False, 
                   separate_targets=False, target_list=None):
    """
    Isolate the data from the targets and return these two objects.
    
    Parameters
    ----------
    
    as_ndarray = True : return as numpy arrays
    
    separate_targets = True : return the targets as four separate objects
                       use this assignement in this case:
                       X, (y1, y2, y3, y4) = isolate_target(df, separate_targets=True)
    """
    
    target_list = target_list or ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

    targets = dataframe[target_list].copy()
    data = dataframe.drop(target_list, axis=1)
    
    if return_arrays:
        data, targets = (df.values for df in (data, targets))
    
    if separate_targets:
        targets = tuple((sr for _, sr in targets.items()) if type(targets).__name__ == 'DataFrame' else targets.T)
    
    return data, targets
    


def categorize(target, test_size=None):
    """
    Bin an array and return the categories.
    Helper function. The output is to be used for the 'stratify' parameter
    of the sklearn.model_selection.train_test_split function.
    """
    def _get_breaks(arr, n_bins):
        q = np.linspace(0,1, num=n_bins+1)
        breaks = sorted(set(np.quantile(arr, q, method='weibull')))
        breaks[-1] += 0.9
        return breaks
    
    #TODO
    if np.array(target).ndim > 1:
        raise NotImplementedError("Multioutput target is note yet implemented")
    
    arr = target
    
    test_size = test_size or 0.25
    soll = round(1 / test_size)  # ceil
    
    for n_bins in range(2, len(arr)):
        breaks = _get_breaks(arr, n_bins)
        
        cats = list(pd.cut(arr, bins=breaks,  
                           include_lowest=True, right=False,
                           labels=range(len(breaks)-1)))
        ist = min(Counter(cats).values())

        if ist < soll:
            n_bins -= 1
            breaks = _get_breaks(arr, n_bins)
            cats = list(pd.cut(arr, bins=breaks, 
                               include_lowest=True, right=False,
                               labels=range(len(breaks)-1)))
            mask = np.isnan(cats)
            cats = np.array(cats)
            cats[mask] = np.nanmax(cats)
            assert len(cats) == len(target), "err"
            return list(cats)



def separate(data, encode=False, target_list=None, return_arrays=False,
             stratify=True, test_split=True, random_state=None):
    """
    Sdeparates the data for our Protein Pathfinders modeling.
    If target_list is not the default list of target column names then
        the data must be a df containing the custom targets (e.g. 16 targets)
    Returns (X_trains, X_tests, Y_trains, Y_tests) where each of them is an array of k objetcs
        where k = number of targets (in our case 4 or 16)
    """
    target_list = target_list or ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]
    
    # encode upd23b_clinical_state_on_medication
    column = "upd23b_clinical_state_on_medication"
    if encode and column in data.columns:
        mapping = encode if type(encode) is dict else {'On': 1, 'Off': 0, np.nan: -1}
        data[[column]] = data[[column]].replace(mapping, inplace=False)  # or like this?: data[column] = data[column].replace(mapping)

    XX = [data.dropna(axis=0, subset=col).drop(sorted(set(target_list).difference({col})), axis=1) for col in target_list]
    YY = [X.pop(col) for X,col in zip(XX, sorted(target_list))]
    
    f = lambda obj: obj.values if return_arrays and hasattr(obj, 'values') else obj
    
    splits = tuple([f(obj) for obj in objects] for objects in (XX,YY))
    
    if test_split:
        test_size = 0.25 if test_split is True else float(test_split)
        splits = np.array([train_test_split(X, Y, stratify=categorize(Y, test_size), 
                                            test_size=test_size, random_state=random_state) 
                           for X,Y in zip(*splits)], dtype=object).T
    
    return splits  # (X_trains, X_tests, y_trains, y_tests) bzw (X_new, y_new)



def _get_column_indeces(columns, data):
    """
    Helper function that returns columns indeces
    
    ignore error = because if for example ColumnSubsetter drops "vistit_month"
    then this block of code would throw an error 
    """
    IGNORE_ERROR = True
    
    if any(type(e) is str for e in columns):
        if not hasattr(data, 'columns'):
            raise TypeError("Your column names contain strings while the data is not a dataframe")
        
        # the following is supposed to work both for a list of strings and for a single string
        columns_, = np.nonzero(np.isin(data.columns.values, columns))
        
        if len(columns_) != len(columns):
            #warn(f"One or more columns from your list has not been identified in the dataframe: {columns}", category=Warning)
            if not IGNORE_ERROR:
                raise ValueError(f"Bad column name(s). One or more provided column names are invalid: {columns}")
        columns = columns_
    return columns



class ColumnSubsetter(BaseEstimator, TransformerMixin):
    """
    Drops columns based on conditions as follows:
        threshold : drop the columns whose proportion of missing values is 
        more than the specified amount.
        threshold = None : do not drop any columns
        
        drop_columns : drop the specified column(s)
        
    Note: if 'threshold' is not None and 'drop_columns' is not None
          then the nan-based column elimination is applied first
          and then the elimination of columns based on 'drop_columns'
    """
    
    def __init__(self, threshold=None, drop_columns=None):
        self.threshold = threshold
        self.drop_columns = drop_columns
    
    def fit(self, X, y=None):
        ### Validate Parameters ###
        # validate threshold
        value = self.threshold

        if hasattr(value, 'rvs'):
            value = value.rvs()
        elif not ((value is None) or (0 < value < 1.0)):
            raise ValueError(f"'threshold' must be a float (0.0, 1.0) or None. Got {value}")
        threshold = float(value or 1)
        
        # validate drop_columns
        value = self.drop_columns
        if not (type(value) in (type(None), int, str, list, tuple) or hasattr(value, '__iter__')):
            raise TypeError("'drop_value' must be None or iterable")
        # restructure the temp value suitable for the mechanics in the algorithm 
        ls = [value] if type(value) in (str, int) else [] if value is None else list(value)

        if len(set(type(e) for e in ls)) not in (0,1):
            raise TypeError("Expected an array of all integers or all strings")
        self.drop_columns = ls
        
        ### Fit ###
        # make a mask based on the percentage of missing values in the columns
        if self.threshold is None:
            # do not estimate the proportion of nan's
            self.__mask = np.ones(X.shape[1]).astype(bool)
        else:
            # assume it is a df
            try:
                # compute the proportions of missing values in each column
                pp = (X.isnull().sum(axis=0) / len(X)).values
            # otherwise assume it is an ndarray
            except AttributeError:
                try:
                    X = X.astype(float)
                except ValueError as err:
                    raise ValueError("Your numpy-array must be converted to the float-type "
                                     "in order to count the missing values, but it is impossible to do so because "
                                     "your input array probably contains some values "
                                     "which cannot be converted to floats. " + (str(err)))
                # compute the proportions of missing values in each column
                pp = np.isnan(X).sum(axis=0) / len(X)
            
            # create a mask to subset columns later
            self.__mask = pp < threshold
        
        # modify the mask based on the user's 'drop_columns'  ('drop_columns' is a list)
        self.drop_columns = _get_column_indeces(self.drop_columns, X)
                    
        # update self.__mask with self.drop_columns
        self.__mask[sorted(set(self.drop_columns))] = False
        return self
    
    def transform(self, X):
        # check for shape mismatch
        if X.shape[1] != len(self.__mask):
            raise ValueError(f"Shape mismatch. Expected {len(self.__mask)} columns. Got {X.shape[1]}")
        
        # is X a df or nd ?
        try:
            data = X.iloc
        except AttributeError:
            data = X
        return data[:, self.__mask].copy()



class Encoder(BaseEstimator, TransformerMixin):
    """
    Encodes upd23b_clinical_state_on_medication
    """
    def __init__(self, column_to_encode=None, copy=True):
        self.copy = copy
        self.column_to_encode = column_to_encode
        
    def fit(self, X, y=None):
        if self.column_to_encode is None:
            self.__col = -1 if type(X) is np.ndarray else "upd23b_clinical_state_on_medication"
        else:
            self.__col = self.column_to_encode
        
        assert type(self.__col) is {np.ndarray: int, pd.DataFrame: str}.get(type(X)), "err"

        self.__mapping = {'On': 1, 'Off': 0, np.nan: -1}
        return self
    
    def transform(self, X, y=None):
        if hasattr(X, 'columns') and self.__col not in X.columns:
            return X
        if type(X) is np.ndarray and not set(X[:,-1]).intersection({'Off', 'On'}):
            return X
        
        data = X.copy() if self.copy else X
        
        if type(data) is np.ndarray:
            data[:, self.__col] = pd.Series(data[:, self.__col])\
                .replace(self.__mapping, inplace=False).values
            data = data.astype(np.float_)
        else:
            data[self.__col] = data[self.__col].replace(self.__mapping, inplace=False)
        return data


    
class Imputer(BaseEstimator, TransformerMixin):
    """
    Custom imputer
    Returns a df it the input was a df (unlike sklearn's native Imputers')
    """
    
    def __init__(self, strategy="median"):
        self.strategy = strategy
        
    def fit(self, X, y=None, **kwargs):
        strategy = str(self.strategy).lower()
        
        if strategy not in ("mean", "median"):
            raise ValueError(f"Bad imputation strategy: {self.strategy}")
        
        self.__imputer = SimpleImputer(strategy=strategy, **kwargs).fit(X)
        return self
    
    def transform(self, X, y=None):
        flag = type(X) is pd.DataFrame
        if flag:
            index = X.index
            columns = X.columns
        
        data = self.__imputer.transform(X)
        
        if flag and type(data) is not pd.DataFrame:
            data = pd.DataFrame(data, columns=columns, index=index)
        
        return data
    
    
class ElementwiseTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms the data element-wise, e.g. natural log applied on each cell
    
    Parameters
    ----------
    function : None or a vectorized function
    
    skip_columns : a list of integers denoting the columns onto which the transformation must not be applied
    """
    
    def __init__(self, function=None, skip_columns=None, copy=True):
        self.function = function
        self.skip_columns = skip_columns
        self.copy = copy
    
    def fit(self, X, y=None):
        ### Parameter Validation ###
        # function validation
        value = self.function
        mapping = {'log': np.log, 'sqrt': np.sqrt, 'log10': np.log10, 'ln': np.log}
        value = mapping.get(value, value)
        
        if type(value).__name__ not in ('ufunc', 'vectorize', 'NoneType'):
            raise TypeError("Must be a vectorized function or None")
        self.function = value
        
        # validate skip_columns
        value = self.skip_columns
        columns = list([] if value is None else [value] if type(value) in (str, int, np.int_) else value)
        self.skip_columns = _get_column_indeces(columns, X)
        
        ### Fit ###
        self.__mask = np.ones(X.shape[1]).astype(bool)
        self.__mask[sorted(set(self.skip_columns))] = False
        return self
    
    def transform(self, X):
        if self.function is None:
            return X
        
        if self.copy:
            X = X.copy()
        
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore')
            
            if type(X) is np.ndarray:
                X[:, self.__mask] = np.nan_to_num(self.function(X[:, self.__mask]), nan=0.0, posinf=0.0, neginf=0.0)
            else:
                X.iloc[:, self.__mask] = self.function(X.iloc[:, self.__mask]).replace({np.nan: 0, np.inf: 0, -np.inf: 0})
        return X
        

class Scaler(BaseEstimator, TransformerMixin):
    """
    Custom Scaler with the functionality specific for our taks:
        The usual way would be to use sklearn's ColumnTransformer
        and direct the data through two separate pipelines for scaling.
        This is an alternative implimentation to use a single Scaler 
        and have it apply two different scaling technique to 
        the protein/pepdide columns on the one hand
        and to the visit_month column on the other.
    
    This Scaler applies StandrdScaler to all columns except those specified
    in the 'special_columns' list. 
    
    Parameters
    ----------
    special_columns = [0, -1] (for example)
    
    special_scaler = "minmax" (for example)
    
    special_scaler = None
        do not apply any scaler to the columns specified in 'special_columns'
    
    If special_columns is not None AND special_scaler = None
        ...
    """
    
    def __init__(self, special_scaler=None, special_columns=None, copy=True):
        self.special_scaler = special_scaler
        self.special_columns = special_columns
        self.general_scaler = StandardScaler()
        self.copy = copy
    
    def fit(self, X, y=None):
        ### Validate Parameters ###
        
        # validate special_columns
        value = self.special_columns
        
        if not (type(value) in (type(None), int, str, list, tuple) or hasattr(value, '__iter__')):
            raise TypeError("'special_columns' must be None or iterable")
        # restructure the temp value suitable for the mechanics in the algorithm 
        ls = [value] if type(value) in (str, int) else [] if value is None else list(value)

        if len(set(type(e) for e in ls)) not in (0,1):
            raise TypeError("Expected an array of all integers or all strings")
        self.special_columns = ls
        
        # validate special_scaler
        mapping = {"minmax": MinMaxScaler, "standard": StandardScaler}
        value = self.special_scaler
        
        if type(value) is str:
            v = value.lower().replace('_', '')
            k = ([k for k in mapping.keys() if k[:6]==v[:6]] + [None])[0]
            if k is None:
                raise ValueError("the 'special_scaler' is not understood")
            self.special_scaler = mapping[k]()
        elif value is None:
            self.special_scaler = None
        elif not hasattr(value, 'transform'):
            raise TypeError("The passed in scaler object must conform to the sklearn's Scaler-classes")
        else:
            self.special_scaler = value() if type(value) is type else value

        ### Fit ###
        # create a mask for columns
        self.__mask = np.ones(X.shape[1]).astype(bool)
        
        column_indeces = _get_column_indeces(self.special_columns, X)
        self.__mask[sorted(set(column_indeces))] = False
        
        # is X an nd or df?
        data = X if not hasattr(X, 'iloc') else X.iloc
        
        # fit scalers
        self.general_scaler.fit(data[:, self.__mask])
        
        if self.special_scaler and sum(~self.__mask):
            self.special_scaler.fit(data[:, ~self.__mask])
        return self
    
    def transform(self, X, y=None):
        data = X
        
        flag = type(data) is pd.DataFrame
        if flag:
            columns = data.columns
            index = data.index
        
        if self.copy:
            data = data.copy()
        
        # assign new values to the nd/df
        try:
            data.iloc[:, self.__mask] = self.general_scaler.transform(data.iloc[:, self.__mask])
        except AttributeError:
            data[:, self.__mask] = self.general_scaler.transform(data[:, self.__mask])
            
        # speciual scaler
        if self.special_scaler and sum(~self.__mask):
            try:
                data.iloc[:, ~self.__mask] = self.special_scaler.transform(data.iloc[:, ~self.__mask])
            except AttributeError:
                data[:, ~self.__mask] = self.special_scaler.transform(data[:, ~self.__mask])
        
        # return a df if the input X was a df
        if flag and type(data) is not pd.DataFrame:
            data = pd.DataFrame(data, columns=columns, index=index)
        
        return data
        

        
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom made feater selector
    
    PCA, Lasso, DT, FR, 
    
    selector = must be a sklearn class initialized OR column indeces ?
                e.g. PCA(0.9), Lasso(),               [0,1,606]
                
    If an estimator is provided then sklearn.feature_selection.SelectFromModel
    is used 
    """
    
    def __init__(self, selector=None, n_components=None):
        self.selector = selector
        self.n_components = n_components
    
    def fit(self, X, y=None):
        # if PCA
        if str(self.selector).upper() == 'PCA':
            self.__selector = PCA(self.n_components).fit(X)
        
        # if indexes of best features
        elif hasattr(self.selector, '__len__') and type(self.selector) is not str:
            raise NotImplementedError
        return self
    
    def transform(self, X, y=None):
        if str(self.selector).upper() == 'PCA':
            return self.__selector.transform(X)
        return X



class ProteinPathfinders(BaseEstimator):
    """
    Make a model that encapsulates multiple sklearn estimators/pipelines.
    """
    def __init__(self, estimators=None):
        self.estimators = estimators
        self.is_fitted = self._is_fitted()
        self.__target_names = None
        
    def _is_fitted(self):
        if not self.estimators:
            self.is_fitted = False
            return False
        for e in self.estimators:
            try: check_is_fitted(e)
            except NotFittedError:
                return False
        self.is_fitted = True
        return True
        
    def fit(self, X, Y):
        if not self.estimators:
            raise RuntimeError("Unable to fit because no estimators were provided")
            
        # remember the column names of the targets
        self.__target_names = list(Y.columns) if hasattr(Y, 'columns') \
            else [sr.name for sr in Y] if all(hasattr(e, 'name') for e in Y) \
                else None
        
        # is it a "matrix" (i.e. nd/df) or an array of objects
        is_matrix = np.array(Y).ndim > 1
        
        XX = [X]*len(self) if is_matrix else X
        Y = Y.values if type(Y) is pd.DataFrame else Y
        
        # what if len(estimators) != Y.shape[1]  
        if is_matrix and hasattr(Y, 'shape') and Y.shape[1] != len(self):
            yy = [Y]*len(self)
        else:
            yy = list(Y.T) if is_matrix else Y

        # do not delete these asserts
        assert len(XX) == len(yy) == len(self), "err1"
        assert all(len(XX[k]) == len(yy[k]) for k in range(len(self))), "err2"
        
        # fit each estimator
        self.estimators = [e.fit(X,y) for e,X,y in zip(self.estimators, XX, yy)]
        self.is_fitted = True
        return self
    
    def predict(self, X, Y=None):
        if not self.is_fitted:
            raise NotFittedError("The model is not yet fitted")
            
        is_matrix = self._is_matrix(X,Y)
        
        XX = [X]*len(self) if is_matrix else X
        
        predictions = np.array([e.predict(X) 
                                for e,X in zip(self.estimators, XX)], 
                               dtype=object)

        if predictions.ndim == 3:
            predictions = np.hstack(predictions).astype(float)
        elif len(set(y.shape for y in predictions)) == 1:
            predictions = predictions.astype(float).T
        
        # restore colnames and index if possible
        if is_matrix and self.__target_names:
        #if self.__target_column_names:
            index = X.index if hasattr(X, 'index') else None
            columns = self.__target_names if len(self.__target_names) == len(self) else\
                      self.__target_names * len(self)
            predictions = pd.DataFrame(predictions, columns=columns, index=index)
        
        elif self.__target_names:
            predictions = [pd.Series(y, name=s, 
                                     index=X.index if hasattr(X, 'index') else None) 
                           for y,s,X in
                           zip(predictions, self.__target_names, XX)]   
        return predictions
    
    def _is_matrix(self, X, Y=None):
        return not(type(X) in (list, tuple) or X.ndim == 1)
    
    def _metrics_helper(self, func, X, Y, **kwargs):
        Y_true, Y_pred = Y, self.predict(X)
        assert len(Y_true) == len(Y_pred), "err"
        if self._is_matrix(X):
            return func(Y_true, Y_pred, **kwargs)
        else:
            return sum(func(ytrue, ypred, **kwargs) 
                       for ytrue,ypred in zip(Y_true, Y_pred)) / len(Y_true)
        
    def smape(self, X, Y, **kwargs):
        return self._metrics_helper(smape_score, X, Y, **kwargs)

    def smape_classic(self, X, Y):
        return self._metrics_helper(smape_score, X, Y, factor=1, add_one=False)

    def rmse(self, X, Y):
        return self._metrics_helper(rmse_score, X, Y)
    
    def r_squared(self, X, Y):
        return self._metrics_helper(r2_score, X, Y)
          
    def print_metrics(self, X, Y, title=None):
        if not self.is_fitted:
            print("Model not fitted yet")
            return
        print()
        if title is not None:
            print(f"overall metrics for {title}:".upper())
        metrics = {"SMAPE (classic)": self.smape_classic(X,Y), 
                   "SMAPE (adjusted)": self.smape(X,Y), 
                   "RMSE": self.rmse(X,Y), 
                   f"R{chr(0x00B2)}": self.r_squared(X,Y)}
        for k,v in metrics.items():
            print(f"{k} = {v:.4}")
    
    @classmethod
    def from_sklearn_objects(cls, objects):
        if not all(isinstance(obj, BaseEstimator) for obj in objects):
            raise TypeError("Bad types")  
        estimators = [e.best_estimator_ if hasattr(e, 'best_estimator_') else e
                      for e in objects]
        new = cls(estimators)
        new.is_fitted = new._is_fitted()
        return new
    
    @classmethod
    def from_hyperparameters(cls, list_of_dicts, pipeline_steps):
        estimators = [clone(Pipeline(pipeline_steps)).set_params(**d) for d in list_of_dicts]
        new = cls(estimators)
        new.is_fitted = False
        return new
    
    def save(self, path=None):
        """
        Saves the entire model.
        Returns the full path to the file.
        """
        # validate the path / assign default file name
        if not isinstance(path, (type(None), str)):
            raise TypeError("'path' should be a sting or None")
        
        path = path or os.getcwd()
        
        if not os.path.isdir(path):
            path, file_name = os.path.split(path)
            if path and not os.path.isdir(path):
                raise FileNotFoundError("path not found")
        else:
            timestamp = datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M_%S")
            file_name = "protein_pathfinders_model_" + timestamp
        full_path = os.path.join(path, file_name)
        
        # save
        with open(full_path, mode='wb') as file:
            pickle.dump(tuple(self.estimators), file)
        return full_path
        
    @classmethod
    def from_file(cls, path):
        with open(path, mode='rb') as file:
            estimators = pickle.load(file)
        new = cls(list(estimators))
        new.is_fitted = new._is_fitted()
        return new
    
    def load(self, path):
        """Wrapper around cls.from_file"""
        new = self.from_file(path)
        self.estimators = list(new.estimators)
        self.is_fitted = new.is_fitted
        del new
        return self
    
    def update(self, estimator: "estimator number (one-based)", 
               hyperparameters_dict):
        self.estimators[estimator-1].set_params(**hyperparameters_dict)
        self.is_fitted = False
        return self
    
    def __len__(self):
        return len(self.estimators or [])
    
    def __getitem__(self, idx):
        return self.estimators[idx]
    
    def __bool__(self):
        return self.is_fitted and bool(self.estimators)
    
    def __eq__(self, other):
        return NotImplemented
    
    def __add__(self, other):
        if not isinstance(other, BaseEstimator):
            return TypeError("unable to add a non sklearn object")
        return self.__class__(self.estimators + [other])
    
    def __radd__(self, other):
        if not isinstance(other, BaseEstimator):
            return TypeError("unable to add a non sklearn object")
        return self.__class__([other] + self.estimators)   
    
    def __iadd__(self, other):
        if not isinstance(other, BaseEstimator):
            return TypeError("unable to add a non sklearn object")
        self.estimators.append(other)
        self.is_fitted = self._is_fitted()
        return self
    
    def __iter__(self):
        return iter(self.estimators)
    
    def __setitem__(self, index, item):
        raise NotImplementedError("not implemented yet")



class AdHockExponentialDistribution:
    def __init__(self, min=None, max=None):
        self.__space = np.logspace(min, max, num=max-min+1, base=10)
    
    def rvs(self, *args, **kwargs):
        return choice(self.__space)
    


class RandomHiddenLayerSizes:
    def __init__(self, n_input_features=None, max_n_layers=None):
        self.max_n_layers = max_n_layers
        self.n_input_features = n_input_features
        
    def rvs(self, *args, **kwargs):
        n = self.n_input_features
        l = randint(2, self.max_n_layers)
        n = randint(int(n*(2/3)), n*2)
        squares = [2**p for p in range(int(log2(n)), 2, -1)]
        nn = sorted( [(random() / (i+1)) *  log(l) * 3
             for i,u in enumerate(squares)], reverse=True)
        nn = [ceil(n) for n in nn]
        uu = sum([[u]*n for u,n in zip(squares, nn)], [])
        return tuple(([n]+uu)[:l])
    
    
    
class Baseline(BaseEstimator):
    """
    Baseline model. Drops missing values. Computes median/mean
    """
    def __init__(self, strategy='median'):
        self.strategy = strategy
    
    def fit(self, X=None, Y=None):
        assert type(Y) is pd.DataFrame, "Y must be a df"
        self.columns = Y.columns
        f = Y.median if self.strategy == 'median' else Y.mean
        self.parameters = f(axis=0, skipna=True)
        return self
    
    def predict(self, X):
        Y_pred = np.array([list(self.parameters)] * len(X))
        return Y_pred if type(X) is np.ndarray else pd.DataFrame(Y_pred, columns=self.columns, index=X.index)
    
    def score(self, X, Y, **kwargs):
        Y_true, Y_pred = Y, self.predict(X)
        y_true = Y_true.values.ravel()
        y_pred = Y_pred.values.ravel()
        mask = np.bitwise_not(np.isnan(y_true))
        y_true, y_pred = (nd[mask] for nd in (y_true, y_pred))
        assert len(y_true) == len(y_pred), "must be of eq length"
        return smape_score(y_true, y_pred, **kwargs)
    

    
## quick and dirty report printing functions ##
def print_report(results,
                 X_train=None, y_train=None,
                 X_test=None, y_test=None, 
                 target=None):
    
    if X_test is None or y_test is None:
        raise ValueError("X_test and y_test must be provided")
    
    rs = results[-1] if type(results) in (list, tuple) else results
    k = len(results) if type(results) in (list, tuple) else target or '?'
    
    print(f"\n\n ***** TARGET {k} *****\n")
    print(f"{len(rs.cv_results_['params'])} models were fitted")
    print("best estimator:", rs.best_estimator_.steps[-1][-1].__class__.__name__)
    print("best parameters:", rs.best_params_); print()
    
    from collections import Counter
    
    try:
        ls = sorted(zip(rs.cv_results_['mean_test_score'], (d['estimator'].__class__.__name__ for d in rs.cv_results_['params'])), 
           key=lambda t: t[0], reverse=True)[:5]
        top = [k for k,v in sorted(Counter([t[-1] for t in ls]).items(), key=lambda t: t[-1], reverse=True)]
        print(f"top estimators for target {k}:", str(top).replace("'", '')[1:-1]); print()
    except (AttributeError, KeyError):
        pass
    

    # fool proof y_test
    try:
        y_test = y_test if y_test.ndim == 1 else y_test[:,k] if type(y_test) is np.ndarray else y_test.iloc[:,k]
    except:
        pass
    
    est = rs.best_estimator_
    
    ## train ##
    if X_train is not None and y_train is not None:
        scorer = make_scorer(smape_score, greater_is_better=False)
        scores = cross_val_score(est, X_train, y_train, scoring=scorer, cv=5)
        smape, std = -scores.mean(), scores.std()
        print(f"CV SMAPE on train set: {smape:.0f}%, std = {std:.2f}")
        
        y_pred = est.predict(X_train)
        smape = smape_score(y_train, y_pred)
        rmse = rmse_score(y_train, y_pred)
        print(f"SMAPE on train set: {smape:.0f}%")
        print("RMSE on train set: ", rmse.round(2))
    else:
        print("report on train set is not available")
    print()
    
    ## test ##
    if X_test is not None and y_test is not None:
        y_pred = est.predict(X_test)
        smape_classic = smape_score(y_test, y_pred, factor=1, add_one=False)
        smape = smape_score(y_test, y_pred)
        rmse = rmse_score(y_test, y_pred)
        print(f"SMAPE (classic) on test set: {smape_classic:.0f}%")
        print(f"SMAPE (adjusted) on test set: {smape:.0f}%")
        print("RMSE on test set:", rmse.round(2))
    else:
        print("report on test set is not available")
    print("_"*80)
    

def print_overall_report(results,
                 X_train, Y_train,
                 X_test, Y_test):
    if None in results or Y_test.shape[1] > len(results):
        print("Unable to produce overall report")
        return
    
    try:
        cv_smape = round(sum(-rs.best_score_ for rs in results) / len(results))
    except AttributeError:
        cv_smape = '?'
    
    results = [e.best_estimator_ if hasattr(e, 'best_estimator_') else e for e in results]
    X_train, Y_train, X_test, Y_test = (np.array(e) for e in (X_train, Y_train, X_test, Y_test))
        
    print("\n***** OVERALL RESULTS *****")
    
    def _print_report(which="?"):
        X,Y = (X_train, Y_train) if which.lower() == "train" else (X_test, Y_test)
        print(f"{which}:".upper())

        if which.lower() == "train":
            print(f" CV SMAPE:       {cv_smape}%")

        Ypred = np.array([e.predict(X) for e in results]).T
        
        assert Ypred.shape==Y.shape and len(X)==len(Ypred), "err2"
        smape = smape_score(Y, Ypred)
        rmse = rmse_score(Y, Ypred)
        
        print(f" SMAPE:          {smape:.0f}%")
        print(f" RMSE:           {rmse:.2f}")
        smape = smape_score(Y, Ypred, factor=1, add_one=False)
        print(f' SMAPE(classic): {smape:.0f}%')
        rsq = sum(e.score(X, Y[:,k]) for k,e in enumerate(results)) / len(results)
        print(f" R^2:            {rsq:.2f}\n")
    
    for which in ("train", "test"):
        _print_report(which)
    
    
 
def print_report_best_models(results):
    for k in range(len(results)):
        d = results[k].cv_results_
        
        scores = np.nan_to_num(d['mean_test_score'], nan=-9999, posinf=-9999, neginf=-9999)
        stds = np.nan_to_num(d['std_test_score'], nan=9999, posinf=9999, neginf=9999)
        
        models = sorted(zip(scores, stds, d['params']), 
                        key=lambda t: (round(-t[0]), t[1]), reverse=False)

        b = 5 # best models (for a particular target)

        print(f"\n\n{'*'*b} best models for target {k+1} {'*'*b}\n".upper())
        for smape, std, params in models[:b]:
            print(f"CV SMAPE = {-smape: .1f}%, SDT = {std: .2f}")
            print(params, end="\n\n")
            
