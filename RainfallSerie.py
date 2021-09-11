import  os
from datetime import datetime
from datetime import timedelta
from collections import OrderedDict
from DPSRIRaster import DpsriRaster

import pandas as pd
import numpy as np
import pmdarima as pm
import itertools
import copy
import sys
import georasters as gr
from work_folder.utils import empty_directory, find_replacement
from multiprocessing import Pool,Manager,Process,Queue

cpu_count = os.cpu_count()


"""
DPSRI SERIES EXTRACTION PROCESS:
1) from the directory containing the generated tiff files by rainbow, we read all the exixting 
dates,and create dictionnary [datetime --> [directory, filename]],
this dictionary is time ordered.
and we have created another dictionnary for step datetime assimilation.
Each dpsri raster in the directory is cropped according to the given original dem downloaded from Nasa,
then as the dpsri have lower resolution than the em file, the dperi file is upsampled to much the n of cells 
in the height and the width.
then the upsampled dpsri is cropped to match the desired extent, either for the entered coordinates, or the shapefile.
then the dpsri raster is converted to csv.
and merged with other csvs.
Then we did replacement in the x and y columns of the dpsri series csv file, because the floating point numbers is different 
between the dem and dpsri rasters.

"""
class RainfallSerie(object):

    def __init__(self,directory,dem,input_extent,working_directory,timesteps,time_lag,tick_time):

        self.directory=directory
        self.timesteps = timesteps
        self.dem = dem
        self.input_extent = input_extent
        self.working_directory = working_directory
        self.dpsri_files=self.get_dpsri_files(time_lag,tick_time)
        self.get_delta_t()
        self.dpsri_csvs={}
        self.rainfall_series=pd.DataFrame()
        self.get_no_data()




    def get_delta_t(self):
        self.time_ordered_files = OrderedDict(sorted(self.dpsri_files.items()))
        time_ordered_keys=list(self.dpsri_files.keys())
        delta_t = time_ordered_keys[1] - time_ordered_keys[0]
        delta_t_min = delta_t / 60
        print("this is inside rainfall series generator:")
        print("delta t in min:")

        min = delta_t_min.seconds
        print(min)
        self.delta_t_min=min
        self.delta_t=delta_t
        self.timesteps_number = 24 * 60 / self.delta_t_min
        time_ordered_files = OrderedDict(sorted(self.dpsri_files.items()))

        if (self.timesteps <= self.files_count):
            keys = list(time_ordered_files.keys())
            i = 0
            while (len(time_ordered_files) > 12):
                print("inside while")
                if (i % 2 == 1):
                    key = keys[i]
                    del time_ordered_files[key]
                i += 1
        self.dpsri_files = time_ordered_files
        self.files_count=len(self.dpsri_files)
        self.forecasting_periods = self.timesteps - self.files_count







    def get_no_data(self):
        for root, dirs, files in os.walk(self.directory):
            for filename in files:
                name, extension = os.path.splitext(filename)
                if (extension == ".tif"):
                    dpsri_raster = DpsriRaster(self.directory + "\\" + filename, self.working_directory)
                    self.no_data=dpsri_raster.no_data
                    break
            break



    def get_dpsri_files(self,time_lag,tick_time):

        dpsri_files = {}
        files_count=0
        delta=timedelta(hours=time_lag)
        start_datetime=tick_time-delta
        end_datetime=tick_time

        for root, dirs, files in os.walk(self.directory):
            for filename in files:
                index = len(root) - 10
                name, extension = os.path.splitext(filename)
                if (extension == ".tif"):
                    ####another condition must be created regarding last modification date

                    year = name[0:4]
                    month = name[4:6]
                    day = name[6:8]
                    hour = name[8:10]
                    min = name[10:12]
                    sec = name[12:14]
                    date = day + " " + month + " " + year + " " + hour + ":" + min + ":" + sec
                    current_datetime = datetime.strptime(date, '%d %m %Y %H:%M:%S')
                    if start_datetime<=current_datetime<=end_datetime:
                        files_count+=1
                        dpsri_files.update({current_datetime: filename})
        self.files_count=files_count
        return dpsri_files


    def process_dpsri_file(self,file,datetime,dpsri_rasters):

        """
        :param path: 
        :param datetime: 
        :param dem_extent: 
        :return: 
        """
        dem_extent=self.dem.original_extent
        input_extent=self.input_extent

        dpsri_raster=DpsriRaster(self.directory+"\\"+file,self.working_directory)

        #dpsri_csv = self.working_directory+r"\temporary_files\dpsri_csv"+"\\"+dpsri_raster.filename+".csv"
        dpsri_raster.reproject()
        dpsri_raster.crop_to_upsample(using_shapefile=False,shape_file=None, minx=dem_extent['minx'], miny=dem_extent['miny'],
                                      maxx=dem_extent['maxx'], maxy=dem_extent['maxy'])
        dpsri_raster.upsample(self.dem.width,self.dem.height)
        dpsri_raster.crop_to_match_extent(using_shapefile=False,shape_file=None,minx=input_extent['minx'],miny=input_extent['miny'],
                                          maxx=input_extent['maxx'],maxy=input_extent['maxy'])

        #rtxyz.translate(dpsri_raster.cropped_to_extent,dpsri_csv)

        folder1 = self.working_directory+"/dpsri_rasters/cropped_for_upsampling"
        folder2=self.working_directory+"/dpsri_rasters/upsampled"

        dpsri_rasters.update({datetime: dpsri_raster.cropped_to_extent})
        empty_directory(folder1)
        empty_directory(folder2)




#####not useddd
    def process_directory(self):
        """
        
        :return: 
        """
        #this for loop need to be paralellized.
        for datetime,file in self.dpsri_files.items():
            self.process_dpsri_file(file,datetime)
        #this for loop need to be paralellized.
        for datetime,file in self.dpsri_csvs.items():
            current = pd.read_csv(file, ",")
            current.rename(columns={'z': datetime}, inplace=True)
            if (self.rainfall_series.empty):
                self.rainfall_series = current
            else:
                self.rainfall_series = pd.merge(self.rainfall_series,current, how="inner", on=['x', 'y'])




    def rainfall_nowcasting(self,rainfall, n_periods):
        """
        
        :param rainfall_serie: pandas dataframe containing rainfall column
        :param n_periods: number of periods to forecast over them
        :return: fc: list of rainfall nowcast over n-periods
        """

        model = pm.auto_arima(rainfall, start_p=1, start_q=1,
                              test='adf',  # use adftest to find optimal 'd'
                              max_p=3, max_q=3,  # maximum p and q
                              m=1,  # frequency of series
                              d=None,  # let model determine 'd'
                              seasonal=False,  # No Seasonality
                              start_P=0,
                              D=0,
                              trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)

        fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
        print("after prediction:")
        print(list(fc))
        return list(fc)

    def func(self, dpsri_files):
        print("thread starting:")


        for datetime, file in dpsri_files.items():
            self.process_dpsri_file(file, datetime,dpsri_files)



    def func2(self,dpsri_rasters):
        time_ordered_files = OrderedDict(sorted(dpsri_rasters.items()))
        for datetime,file in time_ordered_files.items():
            raster = gr.from_file(file)
            current_df = raster.to_pandas()
            print(len(current_df))
            current_df.rename(columns={'value': datetime, 'x': 'Lon', 'y': "Lat"}, inplace=True)
            current_df.drop(columns=["row", 'col'], inplace=True)

            if (self.rainfall_series.empty):
               self.rainfall_series = current_df
            else:
                self.rainfall_series = pd.merge(self.rainfall_series,current_df, how="outer", on=['Lon', 'Lat'])




    def parallelize_dataframe(self,df, func,partitions):

        df_split = np.array_split(df, partitions)
        with Pool(4) as p:
            new_df = pd.concat(p.map(func, df_split))
        return new_df

    def generate_series(self,df):
        last_date = list(self.time_ordered_files.keys())[-1]
        print("timesteps: {0}".format(self.timesteps))
        print("file counts: {0}".format(self.files_count))

        r_sum=df.sum(axis=0, skipna=True)
        r_sum = r_sum.tolist()
        lis=self.remove_zeros(r_sum)

        print(df.columns)

        jk=0
        for col in list(df.columns):
                if col=='Lat':
                    del r_sum[jk]
                if col=='Lon':
                    del r_sum[jk]

                jk+=1

        nr_sum=[ele/len(df) for ele in lis]

        predicted_rainfall = self.rainfall_nowcasting(rainfall=nr_sum, n_periods=self.forecasting_periods)
        rain = [abs(r) for r in predicted_rainfall]
        for i in range(self.forecasting_periods):
                later_date = last_date + (i + 1) * self.delta_t
                df[later_date] = 0

        for index, row in df.iterrows():
                 for i in range(self.forecasting_periods):
                    later_date = last_date + (i + 1) * self.delta_t
                    try:
                        df.at[index,later_date] = rain[i]
                    except Exception as e:

                        print(type(e))  # the exception instance
                        print(e.args)  # arguments stored in .args
                        print(e)

        print(df)
        return  df


    def remove_zeros(self,li):
        i = 0  # loop counter
        length = len(li)  # list length
        while (i < length):
            if (li[i] == 0):
                li.remove(li[i])
                # as an element is removed
                # so decrease the length by 1
                length = length - 1
                # run loop again to check element
                # at same index, when item removed
                # next item will shift to the left
                continue
            i = i + 1
        return li

    def remove_nodata(self,df):

        df.replace(to_replace=self.no_data, value=np.nan, inplace=True)
        df.interpolate(method='linear', axis=1, inplace=True)
        return  df





    def dpsri_processing_directory(self):
        manager = Manager()

        dpsri_dict = manager.dict(self.dpsri_files)

        with Pool(cpu_count) as p:
            p.map(self.func,(dpsri_dict,))
        print(dpsri_dict)


        self.func2(dpsri_dict)



        self.rainfall_series=self.parallelize_dataframe(self.rainfall_series, self.remove_nodata, partitions=4)

        self.time_ordered_files = OrderedDict(sorted(dpsri_dict.items()))
        last_date = list(self.time_ordered_files.keys())[-1]


        self.rainfall_series=self.parallelize_dataframe(self.rainfall_series,self.generate_series,partitions=4)
        #self.rainfall_series.to_csv(csv_path)








"""



num_partitions = 10 #number of partitions to split dataframe
num_cores = 4 #number of cores on your machine

iris = pd.DataFrame(sns.load_dataset('iris'))

    

        
        
        
        
def multiply_columns(data):
    data['length_of_word'] = data['species'].apply(lambda x: len(x))
    return data
    
iris = parallelize_dataframe(iris, multiply_columns)








    
        def dpsri_processing_directory(self):
        manager=Manager()
        dpsri_raster = manager.dict(self.dpsri_files)
        dpsri_csv=Queue()
        with Pool(cpu_count) as p:
            p.map(self.func, (dpsri_raster,dpsri_csv,))


    def parallel_processing_directory(self):
            manager = Manager()
            dpsri_raster_dict=manager.dict(self.dpsri_files)
            dpsri_csv_dict = manager.dict(self.dpsri_csvs)


            with Pool(cpu_count) as p:
               p.map(self.func, (zip(itertools.repeat(dpsri_raster_dict), itertools.repeat(dpsri_csv_dict))))
            print(self.dpsri_csvs)
            with Pool(cpu_count) as p:
                p.map(self.func2,(dpsri_csv_dict,))
                
                
                
        def parallel_processing_directory(self):
        print("Inside parallel processing function:")
        print(self.dpsri_files)
        manager = Manager()
        dpsri_raster_dict = manager.dict(self.dpsri_files)
        dpsri_csv_dict = manager.dict(self.dpsri_csvs)
        jobs = [Process(target=self.func, args=(dpsri_raster_dict,dpsri_csv_dict))
                for i in range(cpu_count)
                ]
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()

        print(dpsri_csv_dict)
        with Pool(cpu_count) as p:
            p.map(self.func2, (dpsri_csv_dict,))

    """







