import numpy
import pandas
import tensorflow
from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


class tsa():
    def __init__(self):

        self.record = None
        self.best_param = None
#         

    def create_dataset(self,dataset,skip=1,lagged=1,forecast=1):
        
        '''
        
        creates custom dataset with lagged X and forecasting Y arrays
        
        Parameters
        ----------
        dataset : array_like 
            1D array or list
            
        skip : integer
            1 (default) skip numbers between each value of dataset
            eg. if skip = 2:
                    [1,2,3], [3,4,5], [5,6,7]
            
        lagged : integer
            1 (default) size of the values of the dataset
            eg. if lagged = 2:
                    [1,2], [3,4], [5,6]
                    
        forecast : integer
            1 (default) number of t+ time to forecast in dataset
                
        
        return :
            x array : X dataset
            y array : y dataset
        
        '''
        
        if skip>forecast:
            raise ValueError("skip value should be less than forecast!")
        
        x, y = [],[]
        n = 0
        m= 1
        
    #     print(lagged,len(dataset)-forecast,skip)
        
        for i in range(lagged,len(dataset)-forecast,skip):
            
            
            x.append([dataset[n:i]])
            y.append([dataset[i:i+forecast]])
            
            n+=skip
        
        
        

        return numpy.array(x,dtype=numpy.float64).reshape(-1,lagged),numpy.array(y,dtype=numpy.float64).reshape(-1,forecast)
        
        
    def forecast_adjusted(self,y_pred,skip,forecast):
        x = []
        for i in range(int(forecast/skip)):
            try:
                x = x+list(y_pred[i][:skip])
            except:
                print('exception occurred')
        return x
    

    def lagged_prediction(self,dataset,val,shape,skip=1):
        
        '''
        removing first values from x and adding val at the end
        
        Parameters
        ----------
        
        dataset : numpy array
            dataset to modify
        
        val : numpy array
            value that is added to the end of the dataset
        
        shape : tuple
            shape of the dataset
        
        skip : integer
            1 (default) skip numbers between each value of dataset
            eg. if skip = 2:
                    [1,2,3], [3,4,5], [5,6,7]
                    
                    
        example :
            x = numpy.array([[1],[2],[3],[4],[5],[6]])
            val = numpy.array([10])
            
            lagged_prediction(x,val,shape(-1,1))
            
            ---> numpy.array([[2],[3],[4],[5],[6],[10]])
        
        
        '''
        
        dataset = list(dataset.reshape((-1)))
        val = list(val.reshape(-1))
        
        
        for i in range(skip):
            dataset.pop(0)
            dataset.append(val[i])

        
        return numpy.array(dataset,dtype=numpy.float64).reshape(shape)




    def lstm_reshape(self,x):
        #  [samples, timesteps, features].
        if numpy.ndim(x)==1:
            x = x.reshape(tuple([1]+[1]+[x.shape[-1]]))
        
        else:
            x = x.reshape(tuple([x.shape[0]]+[1]+[x.shape[-1]]))
        
        return x



    def optimise_model(self,dataset, param,n_train = None,n_test=100,epochs=50):
        
            
        '''
        takes dataset and optimises the parameters
        
        Parameters
        ----------
        
        dataset : numpy_array
            
            dataset to be optimised
            
        
        param : dict
        
            'train_test_split': list
                    a list of values between 0-1 defining train test split. split is not random
                
            'validation_split': list
                    a list of values between 0-1 defining validation split for model validation. split is not random

            'model_layers' : 2d list
                    a list containing [number_of_layers, number_of_LSTM_nodes]
                    
            skip : list
                a list of integer values, skip numbers between each value of dataset
                eg. if skip = 2:
                        [1,2,3], [3,4,5], [5,6,7]

            lagged : list
                a list of integer values. size of the values of the dataset
                eg. if lagged = 2:
                        [1,2], [3,4], [5,6]

            forecast : list
                a list of integer values, number of t+ time to forecast in dataset


            example.
                
                param = {
                        'train_test_split': [0.3],
                        'validation_split': [0.3],
                        'model_layers' : [[1,5],[1,20],[1,40],[1,80],[1,130]], 
                        'skip': [1,10,30],
                        'lagged': [65,130,170], 
                        'forecast': [5,20,30]

                    }

        n_test : int
            100 (default) size of the test data fed into the models
        
        
        return :
            
            param_return : list
                a list containing :
                    param_values : parameter values of the dataset and model
                    y_test : y_true of the model
                    y_hat : y_pred of the model
                    
        
        
        Note : this function requires following custom fuctions :
            
            1) create_dataset
            2) lstm_reshape
            3) lagged_prediction
        
        '''
        
        if len(param['train_test_split'])!=len(param['validation_split']):
            raise ValueError('train_test_split and validation_split should be of same length')
            
        if len(param['skip'])!=len(param['lagged'])!=len(param['forecast']):
            raise ValueError('skip,lagged and forecast should be of same length')
        
        if len(dataset)*param['train_test_split'][0]< numpy.max(param['lagged'])*n_test + numpy.max(param['skip'])*(n_test-1):
            print('size of dataset should be more than',(numpy.max(param['lagged'])*n_test + numpy.max(param['skip'])*(n_test-1))/param['train_test_split'][0])
            print('\nsize of dataset provided : ',len(dataset))
            raise ValueError('uneven test split between models')
            
        if n_train != None:
            if len(dataset)*(1-param['train_test_split'][0]) < n_train:
                raise ValueError('n_train is smaller than X_train data')
            

        
        
        return_param = []
        
        for split_iteration in range(len(param['train_test_split'])):
            
            for data_iteration in range(len(param['skip'])):
                
                for model_iteration in range(len(param['model_layers'])):
                    
                    model_layers = param['model_layers'][model_iteration]
                    test_split = param['train_test_split'][split_iteration]
                    validation_split = param['validation_split'][split_iteration]
                    
                    skip = param['skip'][data_iteration]
                    lagged = param['lagged'][data_iteration]
                    forecast = param['forecast'][data_iteration]
        
                    if skip>forecast:
                        break
        
    #                 n = int(round(len(dataset)*train_test_split))
                    
                    x,y = self.create_dataset(dataset,skip=skip,lagged=lagged,forecast=forecast)

                    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=test_split,shuffle=False)
                    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=validation_split,shuffle=False)


    #                 X_train,y_train = self.create_dataset(dataset[:-n],skip=skip,lagged=lagged,forecast=forecast)
    #                 X_test,y_test = self.create_dataset(dataset[-n:],skip=skip,lagged=lagged,forecast=forecast)
                    
                    X_test = X_test[:n_test]
                    y_test = y_test[:n_test]
                
                    
                    X_train = X_train[:n_train]
                    y_train = y_train[:n_train]

                        
                    
                    X_train,X_test,X_val = self.lstm_reshape(X_train),self.lstm_reshape(X_test),self.lstm_reshape(X_val)
                    
                    
                        
                    
    #                 m = int(round(len(X_train)*validation_split))
                    
    #                 X= X_train[:-m]
    #                 y= y_train[:-m]

    #                 X_val = X_train[-m:]
    #                 y_val =y_train[-m:]
                    

                    callback = EarlyStopping(
                        monitor='loss',
                        min_delta=0.001,
                        patience=3,
                        verbose=1,
                        mode='auto',
                        baseline=None,
                        restore_best_weights=False
                    )

                    model = Sequential()

                    for i in range(model_layers[0]):
                        model.add(layers.LSTM(model_layers[1],return_sequences=False))


                    model.add(layers.Dense(len(y_val[0])))
                    model.compile(loss='mean_squared_error',optimizer='adam')



                    model.fit(x=X_train,y=y_train,validation_data=(X_val,y_val),epochs=epochs,callbacks=callback,workers=-1,use_multiprocessing=True)

                    shape = (1,1,-1)


                    x = self.lstm_reshape(X_test[0])
                    y_hat = []

                    for i in range(len(y_test)):
                        print(x.shape,'\n\n')
                        val = numpy.float64(model.predict(x))
                        try:
                            y_hat.append(list(val[0]))
                        except:
                            y_hat.append([val])
    #                     try:
    #                         y_hat = y_hat+list(val[0][:skip])
    #                     except:
    #                         y_hat = y_hat+list(val[0])
                            
                        x = self.lagged_prediction(x,val,shape,skip=skip)
                    
                    y_test = self.forecast_adjusted(y_test,skip=skip,forecast=len(y_test))
                    y_hat = self.forecast_adjusted(y_hat,skip=skip,forecast=len(y_test))
                    
                    
                    param_values = {
                        'train_test_split': [test_split],
                        'validation_split': [validation_split],

                        'model_layers' : [model_layers], 

                        'skip': [skip],
                        'lagged': [lagged], 
                        'forecast': [forecast]
                        
                    }
                    
                    print('\n\n\n', param_values,'\n\n\n')
                    
                    return_param.append([param_values,y_test,y_hat])
                    
                    
        self.record = return_param  
        return return_param


    def get_record(self):
        '''
        Get the optimised model

        return :

            record : list of models
        '''
        if self.record == None:
            raise KeyError('No model found, please run optimise_model')

        else:
            return self.record


    def get_best_model(self):
        pass

    def get_models_dataframe(self,loss='mse'):
        '''
        returns a dataframe of consisiting the models sorted from best to worse.

        Parameters
        ----------
        loss : str
            valid inputs :

            mse  :  mean squared error
            mae  :  mean absolute error
            rmse :  root mean squared error

        return :
            dataframe
        '''
        def rmse(y_true,y_pred):
            return numpy.sqrt(mean_squared_error(y_true,y_pred))

        if loss == 'mse':
            loss = mean_squared_error()
            loss_str = 'mean_squared_error'
        elif loss == 'mae':
            loss = mean_absolute_error()
            loss_str = 'mean_absolute_error'
        elif loss == 'rmse':
            loss = rmse()
            loss_str = 'root_mean_squared_error'
        else:
            raise ValueError('invalid argument given in loss parameter')
        

        if self.record == None:
            raise KeyError('No model found, please run optimise_model')

        loss_result, model, lag, skip, forecast = [],[],[],[],[]
        for i in self.record:
            try:
                loss_result.append(mean_squared_error(i[1],i[2]))
                model.append(i[0]['model_layers'][0][1])
                lag.append(i[0]['lagged'][0])
                skip.append(i[0]['skip'][0])
                forecast.append(i[0]['forecast'][0])
            except:
                pass
        
        dataframe = pd.DataFrame({'model_layers':model,'lagged':lag,loss_str:loss_result,'skip':skip,'forecast':forecast})

        dataframe.sort_values(by=loss_str,inplace=True)

        return dataframe


    
    def model_dataset(self,dataset,param):

        '''

        given the parameters modifies and returns a suitable dataset for LSTM model

        Parameters
        ----------
        
        dataset : numpy_array 1d
            
            dataset that needs to be made that is suitable for the LSTM model
            
        
        param : dict
        
            'train_test_split': list
                    a list of values between 0-1 defining train test split. split is not random
                
            'validation_split': list
                    a list of values between 0-1 defining validation split for model validation. split is not random

            'model_layers' : 2d list
                    a list containing [number_of_layers, number_of_LSTM_nodes]
                    
            skip : list
                a list of integer values, skip numbers between each value of dataset
                eg. if skip = 2:
                        [1,2,3], [3,4,5], [5,6,7]

            lagged : list
                a list of integer values. size of the values of the dataset
                eg. if lagged = 2:
                        [1,2], [3,4], [5,6]

            forecast : list
                a list of integer values, number of t+ time to forecast in dataset


            example.
                
                param = {
                        'train_test_split': [0.3],
                        'validation_split': [0.3],

                        'model_layers' : [[1,5]], 

                        'skip': [1],
                        'lagged': [30], 
                        'forecast': [1]

                    }
        
        return :
            
            data : dict
                the dictionary named data contains :
                            {'train'      : [X_train,y_train],
                            'validataion' : [X_val,y_val],
                            'test'        : [X_test,y_test]}

        '''
    
        x,y = self.create_dataset(dataset,skip=param['skip'][0],lagged=param['lagged'][0],forecast=param['forecast'][0])
    
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=param['train_test_split'][0],shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=param['validation_split'][0],shuffle=False)
        
        X_train,X_val,X_test = self.lstm_reshape(X_train),self.lstm_reshape(X_val),self.lstm_reshape(X_test)
        
    
        
        data = {
            'train': [X_train,y_train],
            'validataion' : [X_val,y_val],
            'test' : [X_test,y_test]
        }
        
        
        return data








