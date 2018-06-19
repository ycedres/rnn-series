import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class OutputManager(object):

    def __init__(self,output_file=''):
        self._df_prediction = None
        pass

    def save(self,option='file'):
        if option == 'file':
            pass
        if option == 'print':
            print(self._data)

    def set_output_config(self,save,basedir=None,file_prefix=None,horizon=None,
                          input_descriptor_string=None,
                          output_filename=None):
       self._save = save
       self._basedir = basedir
       self._file_prefix = file_prefix
       self._horizon = horizon
       self._output_filename = output_filename
       self._input_descriptor_string = input_descriptor_string

    def print_output(self,data):
        print(data)

    def plot_scatter(self):
        x = self._df_prediction['target']
        y = self._df_prediction['prediction']

        plt.xlabel('prediction')
        plt.ylabel('target')
        plt.scatter(x,y,c='b')

        if self._save:

            directory = self._basedir + '/' + self._file_prefix + '/'
            filename = directory + 'scatter.png'

            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(filename)
        else:
            plt.show(block=False)

    def plot_scatter_diagonal(self,title):
        # df_x = pd.DataFrame(x)
        # df_x.index = y.index
        # df = pd.concat([df_x,y],axis=1)
        # df.columns = ['a','b']
        x = self._df_prediction['target']
        y = self._df_prediction['prediction']

        f, ax = plt.subplots(1,1,figsize=(10,10))
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        ax.set_xlim(x_min+1, x_max+1)
        ax.set_ylim(x_min+1, x_max+1)
        ax.plot((x_min, x_max), (x_min, x_max), lw=3, c='r')
        ax.scatter(x,y,c='b')
        ax.set(xlabel='target: wind speed (m/s)', ylabel='prediction: wind speed (m/s)')
        ax.set_title('RDTR: ' + title)
        #plt.figure()
        # self._df_prediction.plot(ax=ax,
        #         x='prediction',
        #         y='target',
        #         kind='scatter',
        #         c='b'
        #         )

        if self._save:

            directory = self._basedir + '/' + self._file_prefix + '_' + \
            self._input_descriptor_string + '/'

            filename = directory + 'scatter.png'

            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(filename)
        else:
            plt.show(block=False)



    def plot(self,x,y):
        x = pd.DataFrame(x)
        x.index = y.index
        df = pd.concat([x,y],axis=1)

        df.plot(figsize=(15,5),title="RLSTM",legend=False)
        # plt.plot(x,y)
        if self._save:

            directory = self._basedir + '/' + self._file_prefix + '_' + \
            self._input_descriptor_string + '/'

            filename = directory + 'serie.png'

            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(filename)
        else:
            plt.show(block=False)


    def set_df_prediction(self,x:np.array,y:pd.DataFrame):
        df_x = pd.DataFrame(x)
        df_x.index = y.index
        self._df_prediction = pd.concat([df_x,y],axis=1)
        self._df_prediction.columns = ['prediction','target']
        if self._save:

            directory = self._basedir + '/' + self._file_prefix + '_' + \
            self._input_descriptor_string + '/'
            filename = directory + self._output_filename

            if not os.path.exists(directory):
                os.makedirs(directory)

            self._df_prediction.to_csv(filename,sep=';')


    def get_test_mae(self):
        self._df_prediction['mae'] = np.abs(self._df_prediction.prediction -
                                      self._df_prediction.target)
        return  self._df_prediction.mae.sum() / len(self._df_prediction.mae)

    def get_test_mse(self):
        self._df_prediction['mse'] = np.power(self._df_prediction.prediction -
                                       self._df_prediction.target, 2)
        return self._df_prediction.mse.sum() / len(self._df_prediction.mse)

    def get_test_rmse(self):
        return np.sqrt(self.get_test_mse())

    def get_test_r2(self,reference_mse):
        return (reference_mse - self.get_test_mse()) / reference_mse

    def save_error_estimators(self,d):
        directory = self._basedir + '/' + self._file_prefix + '_' + \
        self._input_descriptor_string + '/'
        filename = directory + 'errors.json'

        if not os.path.exists(directory):

            os.makedirs(directory)

        f = open(filename,'w')
        import json
        d = json.dumps(d)
        f.write(d)
        f.close()

    def save_experiment_descriptor(self,experiment_name,features_config,train_config,
                               model_config,errors,description):

        directory = self._basedir + '/' + self._file_prefix + '_' + \
        self._input_descriptor_string + '/'
        filename = directory + 'description.json'

        if not os.path.exists(directory):
            os.makedirs(directory)

        import json
        jsonstr = "{"
        jsonstr += "\"name\": \"" + experiment_name + "\","
        jsonstr += "\"horizon\":" +  str(self._horizon)  + ","
        jsonstr += "\"features_config\":" + json.dumps(features_config) + ","
        jsonstr += "\"train_config\":" + json.dumps(train_config)  + ","
        jsonstr += "\"model_config\":" + json.dumps(model_config)  + ","
        jsonstr += "\"errors\":" + json.dumps(errors) + ","
        jsonstr += "\"description\":" + json.dumps(description)

        jsonstr +=  "}"

        # from pprint import pprint
        # pprint(jsonstr)

        f = open(filename,"w")
        f.write(jsonstr)
