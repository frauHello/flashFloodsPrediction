from datetime import datetime
from datetime import timedelta
"""
current_datetime=datetime.now()
d=timedelta(hours=h)

"""
from DemRaster import DemRaster
from RainfallSerie import RainfallSerie

import pandas as pd
import csv
from XMLObject import XMLObject
import  random
import geopandas as gpd
import os
import sys
import numpy as np
from multiprocessing import Pool

class DatasetGenerator(object):

    def __init__(self,merged_dem,now,past_now,timesteps,dpsri_directory,extent,working_directory,cropped_name):
        self.extent=extent
        self.now=now
        self.timesteps=timesteps
        self.past_now=past_now
        self.dpsri_directory=dpsri_directory
        self.working_directory=working_directory
        self.cropped_name=cropped_name
        self.gis_df=pd.DataFrame()
        self.dem = DemRaster(merged_dem, self.working_directory)


    def gis_task(self):

        self.dem.crop_to_match_extent(self.cropped_name ,False,'',minx=self.extent['minx'], miny=self.extent['miny'],
                                 maxx=self.extent['maxx'], maxy=self.extent['maxy'])
        self.dem.get_dem_correction()
        self.dem.get_upslope_contributing_area()
        self.dem.get_elevation()
        self.dem.get_slope()
        self.dem.get_aspect()
        self.dem.get_plan_curvature()
        self.dem.get_upslope_flow_path_length()
        self.dem.get_wetness_index()
        self.dem.get_distance_to_stream()
        self.dem.get_stream_power_index()
        gis_data=[self.dem.slope_df,self.dem.spi_df,self.dem.flow_length_df,self.dem.wi_df,
                  self.dem.plan_curvature_df,self.dem.aspect_df,self.dem.distance_to_stream_df]
        self.gis_df = self.dem.elevation_df
        print("length of self.dem.elevation_df:{0}".format(self.dem.elevation_df))

        for item in gis_data:
             self.gis_df = pd.merge(self.gis_df,item, how="outer", on=['Lon', 'Lat'])

        print(self.gis_df.columns)
        columns=self.gis_df.columns
        for col in columns:
            self.gis_df[col].fillna((self.gis_df[col].mean()), inplace=True)
        #self.gis_df.to_csv(r"C:\Users\user\Desktop\FlashFloodApplication\work_folder\temporary_files\gis.csv")

    def find_replacement(self,ds1, ds2):

        y2 = ds2['Lat'].to_list()
        x2 = ds2['Lon'].to_list()
        y2 = list(dict.fromkeys(y2))
        x2 = list(dict.fromkeys(x2))
        x1 = ds1['Lon'].to_list()
        x1 = list(dict.fromkeys(x1))
        y1 = ds1['Lat'].to_list()
        y1 = list(dict.fromkeys(y1))
        replacement = {}

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        for i in y2:
            val = find_nearest(y1, i)
            replacement[i] = val
        for i in x2:
            val = find_nearest(x1, i)
            replacement[i] = val
        self.replacement=replacement
        #self.rainfall.rainfall_series.replace(to_replace=replacement, value=None, inplace=True)

        #self.gis_df=self.parallelize_dataframe(self.gis_df,self.replace_lon_lat,4)
        #self.rainfall.rainfall_series=self.parallelize_dataframe(self.rainfall.rainfall_series,self.replace_lon_lat,4)

        #self.dataset=pd.merge(self.rainfall.rainfall_series,self.gis_df, how="inner", on=['Lon', 'Lat'])





    def replace_lon_lat(self,df):
        df.replace(to_replace=self.replacement,value=None,inplace=True)
        return  df


    def rainfall_task(self):
        rainfall=RainfallSerie(directory=self.dpsri_directory, dem=self.dem, input_extent=self.extent,
                               working_directory=self.working_directory,
                               timesteps=self.timesteps, time_lag=self.past_now, tick_time=self.now)
        rainfall.dpsri_processing_directory()### if name==main
        self.rainfall=rainfall
        #self.rainfall.rainfall_series.to_csv(r"C:\Users\user\Desktop\FlashFloodApplication\work_folder\temporary_files\rain.csv")

    def generate_graphs(self):
        num_splits = 1
        duration=self.past_now*60*60
        time_steps=self.timesteps
        print("total_time_steps: {0}".format(time_steps))
        flood_time_list=list(range(1, int(time_steps)+1))

        time_steps = [str(i) for i in flood_time_list]
        ignore_cols = ['FloodOccurence','row','col']

        """
        
        """
        print("the cell col names as retrieved from the csv")
        col_names=['z', 'Lon', 'Lat', 'slope', 'stream power index', 'flow length', 'wetness index', 'plan curvature',
                   'aspect', 'rainfall']


        col_types=['f', 'lon', 'lon', 'f', 'f', 'f', 'f', 'f', 'f', 'f_array']

        schema_path = self.working_directory+r"\experiment\flood\flood_schema.xml"
        graph_path = self.working_directory + r"\experiment\flood\flood_graphs.xml"

        graph_cols=['z', 'Lon', 'Lat', 'slope', 'stream_power_index', 'flow_length', 'wetness_index', 'plan_curvature',
                   'aspect']


        class_labels = self.classLabels(num_splits)
        print("Classes:", len(class_labels), class_labels)
        #self.createSchema(col_names,col_types, class_labels, ignore_cols, schema_path,time_steps)
        self.dataToXML(graph_cols,graph_path,ignore_cols,time_steps)

    def classLabels(self,num_splits):
        """
        :param num_splits:number of classes (2 classes) Binary classification
        :return: list of integers from 0 till num_splits
        """
        return [str(i) for i in range(num_splits + 1)]
    def dataToXML(self,col_names, outfile, ignore,time_steps):

        """

        :param col_names:
        :param reader:
        :param outfile:
        :param ignore:
        :param labeler:
        :param time: Is the time of flood occurrence we will use it start time and end time
        :return:
        """
        ignore = [col.lower() for col in ignore]
        xml = XMLObject('graphs')
        xml.graphs = []


        # convert each row into a graph
        for i, row in self.dataset.iterrows():
            row=row.tolist()
            gis=row[0:len(col_names)]
            rain=row[len(col_names):]

            # row is a list containing the row values
            graph = XMLObject('graph', xml.graphs)
            graph.objects = []
            graph._id = i
            graph._start_time = time_steps[0]
            graph._end_time = len(time_steps)
            graph._flood_time = len(time_steps)####duplicate


            # write each column as an attribute
            cell = XMLObject('cell')
            for col_name, value in zip(col_names, gis):
                if col_name.lower() in ignore:
                    continue
                setattr(cell, col_name, value)

                if col_name == 'Lon':
                    cell_lon = float(value)
                    print("cell_lon: {0}".format(cell_lon))

                if col_name == 'Lat':
                    cell_lat = float(value)
                    print("cell_lat: {0}".format(cell_lat))



            rainfall = [float(k) for k in rain]
            setattr(cell, "rainfall", rainfall)
            cell._id = i
            graph.objects.append(cell)

        xml.graphs.append(graph)
        xml.writeXML(open(outfile, 'w'))
        print('Num Graphs: %s' % i)

    def createSchema(self,col_names, col_types, class_labels, ignore, outfile, flood_time):
        col_type_map = dict(d='Discrete', f='Float', i='Int', s='String', lon='LatLong', f_array='FloatArray')
        ignore = [col.lower() for col in ignore]
        schema = XMLObject('schema')
        schema.class_labels = XMLObject('attrib')
        schema.class_labels._name = 'class_labels'
        schema.class_labels._type = 'str-array'
        schema.class_labels._text = class_labels
        schema.flood_time = XMLObject('attrib')
        schema.flood_time._name = 'flood_time'
        schema.flood_time._type = 'Discrete'
        schema.flood_time._text = flood_time

        schema.graph = XMLObject('graph')
        schema.graph.sample = XMLObject('object')
        schema.graph.sample._type = 'cell'
        schema.graph.sample.attributes = []
        for col_name, col_type in zip(col_names, col_types):
            if col_name.lower() in ignore:
                print("this colnmae is in ignore :{0}".format(col_name.lower()))
                continue
            attrib = XMLObject('attribute', schema.graph.sample.attributes)
            attrib._name = col_name
            attrib._type = col_type_map[col_type.lower()]
            print("the col_name is {0} and the type is {1}".format(col_name, col_type))

        schema.writeXML(open(outfile, 'w'))
    def parallelize_dataframe(self,df, func,partitions):

        df_split = np.array_split(df, partitions)
        with Pool(4) as p:
            new_df = pd.concat(p.map(func, df_split))
        return new_df


def main():
    shp = r"C:\Users\user\Desktop\datasetGeneration\data_Sardinia_Italy\03CAPOTERRA_vector\EMSR329_03CAPOTERRA_GRA_v1_area_of_interest_a.shp"
    cropping_vector = gpd.read_file(shp)
    working_directory=r"C:\Users\user\Desktop\FlashFloodApplication\work_folder\temporary_files"
    minx = cropping_vector.bounds.loc[0, 'minx']
    miny = cropping_vector.bounds.loc[0, 'miny']
    maxx = cropping_vector.bounds.loc[0, 'maxx']
    maxy = cropping_vector.bounds.loc[0, 'maxy']
    extent={"minx":minx,'miny':miny,"maxx":maxx,"maxy":maxy}
    #############################################
    dpsri_directory=r"C:\Users\user\Desktop\dpsri_directory"
    date_time_list=[]


    for root, dirs, files in os.walk(dpsri_directory):
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

                date_time_list.append(current_datetime)

    date_time_list = list(dict.fromkeys(date_time_list))
    time_ordered_files = sorted(date_time_list)
    now=time_ordered_files[len(date_time_list)-1]

    past_now=time_ordered_files[0]


    ds=DatasetGenerator(now=now,past_now=2,timesteps=19,dpsri_directory=dpsri_directory,
                        extent=extent,working_directory=working_directory,cropped_name="first")
    ds.gis_task()
    print("the length of ds.gis_df")
    print(len(ds.gis_df))
    print(ds.gis_df)

    ds.rainfall_task()
    print(len(ds.rainfall.rainfall_series))

    ds.find_replacement(ds.gis_df, ds.rainfall.rainfall_series)
    ds.gis_df=ds.parallelize_dataframe(ds.gis_df,ds.replace_lon_lat,4)
    ds.rainfall.rainfall_series=ds.parallelize_dataframe(ds.rainfall.rainfall_series,ds.replace_lon_lat,4)

    ds.dataset=pd.merge(ds.rainfall.rainfall_series,ds.gis_df, how="inner", on=['Lon', 'Lat'])

    #dataset = pd.merge(ds.rainfall.rainfall_series, ds.gis_df, how="inner", on=['Lon', 'Lat'])
    #ds.merging_ds()
    print(ds.dataset.head())
    print("length of ds.dataset:")
    print(len(ds.dataset))



    ds.generate_graphs()


if __name__== '__main__':
    main()