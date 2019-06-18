def run_round_results(self, out):

    '''Called from logging/logging_run.py

    THE MAIN FUNCTION FOR CREATING RESULTS FOR EACH ROUNDself.
    Takes in the history object from model.fit() and handles it.

    NOTE: The epoch level data will be dropped here each round.

    '''

    self._round_epochs = len(list(out.history.values())[0])

    _round_result_out = [self._round_epochs]

    # use header keys as template, skipping round_epochs
    _header_keys = self.result[0][1:]
    
    for key in _header_keys:
        if key in out.history:
            _round_result_out.append(out.history[key][-1])
    
        elif key in self.round_params:
            _round_result_out.append(self.round_params[key])
    
        else:
            raise KeyError('Header key is not present in round results: ' + key)
    
    # warn if any results were not included
    _missing_history = [r for r in list(out.history.keys()) if r not in _header_keys]
    _missing_params = [r for r in list(self.round_params.keys()) if r not in _header_keys]
    
    if len(_missing_history) > 0:
        print('WARNING history items are being dropped results: %s' % _missing_history)
    
    if len(_missing_params) > 0:
        print('WARNING round_params are being dropped results: %s' % _missing_params)

    return _round_result_out


def save_result(self):

    '''SAVES THE RESULTS/PARAMETERS TO A CSV SPECIFIC TO THE EXPERIMENT'''

    import numpy as np

    np.savetxt(self.experiment_name + '.csv',
               self.result,
               fmt='%s',
               delimiter=',')


def result_todf(self):

    '''ADDS A DATAFRAME VERSION OF THE RESULTS TO THE CLASS OBJECT'''

    import pandas as pd

    # create dataframe for results
    cols = self.result[0]
    self.result = pd.DataFrame(self.result[1:])
    self.result.columns = cols

    return self


def peak_epochs_todf(self):

    import pandas as pd

    return pd.DataFrame(self.peak_epochs, columns=self.peak_epochs[0]).drop(0)
