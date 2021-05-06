#!/usr/bin/env python
# coding: utf-8

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL, space_eval
from Models import MaxConv
from Models import MyModel

params_cnn = {
    'n_conv2D': hp.choice('n_conv2D', range(1, 3)),
    'kernel_size': hp.choice('kernel_size', [(3, 3), (5, 5)]),
    'pool_size': hp.choice('pool_size', [(3, 3), (5, 5)]),
    'activation': hp.choice('activation', ['relu', 'selu']),
    'n_neurons': hp.choice('n_neurons', [4096, 2048])
}

   
epochs = range(100, 301, 100)

default_search_space = { 
    #'optimizer': hp.choice('optimizer', optimizers),
    'epochs': hp.choice('epochs', epochs),
    'cnn': params_cnn,
}

class Optimize:
    
    def __init__(self, name_model='', search_space=None, max_trials=10):
        
        self.name_model = name_model
        self.trials_step = 10  # how many additional trials to do after loading saved trials. 1 = save after iteration
        self.max_trials = max_trials  # initial max_trials. put something small to not have to wait
        self.search_space = search_space

        try:  # try to load an already saved trials object, and increase the max
            self.trials = pickle.load(open(name_model + ".hyperopt", "rb"))
            print("Found saved Trials! Loading...")
            self.max_trials = len(trials.trials) + trials_step
            print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
        except:  # create a new trials object and start searching
            self.trials = Trials()
            
            
    def objective(self, search_space):
        pass
    
    def return_best(self):
        """Function to return the best hyper-parameters"""
        assert hasattr(self, '_best_result') is True, 'Cannot find golden setting. Has optimize() been called?'
        return space_eval(self.search_space, self._best_result)
    
    def fit(self):
       
        self._best_result = fmin(fn=self.objective,
                                 space=self.search_space,
                                 algo=tpe.suggest,
                                 max_evals=self.max_trials,
                                 trials=self.trials)

        print("Best:", self._best_result)
        

        # save the trials object
        with open(self.name_model + ".hyperopt", "wb") as f:
            pickle.dump(self.trials, f)
            
            
        self.results = pd.DataFrame(columns=['iteration'] + columns + ['loss'])
        
        for idx, trial in enumerate(self.trials.trials):
            row = [idx]
            translated_eval = space_eval(search_space, {k: v[0] for k, v in trial['misc']['vals'].items()})
            for k in columns:
                row.append(translated_eval[k])
            row.append(trial['result']['loss'])
            self.results.loc[idx] = row 
            

class Optimize_cnn(Optimize):
    
    def __init__(self, name_model='', search_space=None, max_trials=10):
        
        super().__init__(name_model=name_model, search_space=search_space, max_trials=max_trials)

        global default_search_space
        if search_space is None:
            self.search_space = dict(default_search_space)
        else:
            self.search_space = dict(search_space)
    
    def objective(self, search_space):
        
        if 'cnn' in search_space:
            params = search_space['cnn']
            del search_space['cnn']
        
        model = MyModel(MaxConv(**params), verbose=2, **search_space)
        
        # We train the model
        model.fit(**self.kwargs)
       

        best_epoch_acc = np.argmax(model.history.history['val_accuracy']) 
        val_acc = np.max(model.history.history['val_accuracy']) 
        train_acc = model.history.history['accuracy'][best_epoch_acc] 


        print('Best Epoch : {} - train accuracy {:.2f} - val accuracy: {:.2f}'.format(best_epoch_acc,
                                                                                      train_acc,
                                                                                      val_acc))
        return {'loss': - val_acc,
                'best_epoch': best_epoch_acc,
                'eval_time': time.time(),
                'status': STATUS_OK,
                'model': model,
                'val_accuracy': val_acc}
    
    def fit(self, X=None, y=None, validation_split=0.0, validation_data=None, **kwargs):

            defaults = {'X': X, 'y': y, 'validation_split': validation_split, 'validation_data': validation_data}
            defaults = {k:v for k, v in defaults.items() if k not in kwargs}
            self.kwargs = {**kwargs, **defaults}

            super().fit()