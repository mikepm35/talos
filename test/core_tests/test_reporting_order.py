import time as ti

import talos as ta

from talos.scan.scan_prepare import scan_prepare
from talos.scan.scan_round import scan_round
from talos.scan.scan_finish import scan_finish

from talos.model.ingest_model import ingest_model

from talos.logging.logging_run import logging_run
from talos.logging.logging_finish import logging_finish

from .test_reporting_object import test_reporting_object
from .test_scan import BinaryTest

def test_reporting_order():
    '''
    Tests logic to ensure round reports are correctly aggregated.
    '''
    
    print('Starting test reporting order...')
    
    binary_test = BinaryTest()
    
    class ScanTest():
        def __init__(self):
            self.x = binary_test.x_train
            self.y = binary_test.y_train
            self.params = binary_test.values_list
            self.model = ta.templates.models.cervical_cancer
            self.experiment_name = 'ReportingOrderTest'
            self.x_val = binary_test.x_val
            self.y_val = binary_test.y_val
            self.val_split = .3
            self.random_method = 'crypto_uniform'

            # reducers
            self.performance_target = None
            self.fraction_limit = None
            self.round_limit = 5
            self.time_limit = None
            self.boolean_limit = None

            # reduction related
            self.reduction_method = None
            self.reduction_interval = 50
            self.reduction_window = 20
            self.reduction_threshold = 0.2
            self.reduction_metric = 'val_acc'
            self.minimize_loss = False

            # other
            self.debug = True
            self.seed = 2423
            self.clear_session = False
            self.disable_progress_bar = True
            self.print_params = False

    scan_test = ScanTest()
    
    scan_test = scan_prepare(scan_test)

    # the main cycle of the experiment
    _round_counter = 0
    while True:
        # get the parameters
        scan_test.round_params = scan_test.param_object.round_parameters()

        # break when there is no more permutations left
        if scan_test.round_params is False:
            break
        
        # otherwise proceed with next permutation
        scan_test = scan_round(scan_test)
        
        # set start time
        round_start = ti.strftime('%D-%H%M%S')
        start = ti.time()

        # fit the model
        scan_test.model_history, scan_test.keras_model = ingest_model(scan_test)
        scan_test.round_history.append(scan_test.model_history.history)


        # handle logging of results
        
        # RUN TESTS BY ROUND
        if _round_counter == 1:
            # remove a parameter for a round
            del scan_test.round_params[list(scan_test.round_params.keys())[1]]
            
            try:
                scan_test = logging_run(scan_test, round_start, start, scan_test.model_history)
            except KeyError as e:
                print('Successfully caught KeyError for missing parameter: %s' % e)
            
        elif _round_counter == 2:
            # change the order of the parameters
            _temp_param_key = list(scan_test.round_params.keys())[1]
            _temp_param = scan_test.round_params[_temp_param_key]
            del scan_test.round_params[_temp_param_key]
            scan_test.round_params[_temp_param_key] = _temp_param
            scan_test = logging_run(scan_test, round_start, start, scan_test.model_history)

        elif _round_counter == 3:
            # excess params (warning)
            scan_test.round_params['dummy_param'] = 'dummy_param'
            scan_test.model_history.history['dummy_history'] = 'dummy_history'
            print('Expecting warning for missing parameter key and history key')
            scan_test = logging_run(scan_test, round_start, start, scan_test.model_history)
            
        else:
            scan_test = logging_run(scan_test, round_start, start, scan_test.model_history)      
            
        _round_counter += 1  
        
    # finish
    scan_test = logging_finish(scan_test)
    scan_test = scan_finish(scan_test)    

    # run reporting test on result
    test_reporting_object(scan_test)
        
    return 'Finished testing reporting order'
    
    
    