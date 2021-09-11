import argparse
import math
import rasterio
from rasterio.merge import merge as merge_tool
from random import randint
from datetime import datetime
import pandas as pd
from DatasetGenerator import DatasetGenerator
from work_folder.experiment.ExperimentLoader import loadExperimentFile
import pickle
from utilities.Config import Config


import os
import sys

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("maxx", type=float,
                        help="maxx coordinate in degrees")
    parser.add_argument("maxy", type=float,
                        help="maxy coordinate in degrees")
    parser.add_argument("minx", type=float,
                        help="minx coordinate in degrees")
    parser.add_argument("miny", type=float,
                        help="miny coordinate in degrees")
    parser.add_argument("time", type=str,
                        help="enter the word now or the 24 hour time in the format(day month year h:min:sec)")
    parser.add_argument("dem_directory", type=str,
                        help="SRTM directory absolute path")

    parser.add_argument("dpsri_directory", type=str,
                        help="dpsri directory absolute path")
    parser.add_argument("visualize", type=bool,
                        help="view output on openstreet map")
    args = parser.parse_args()
    maxx=args.maxx
    maxy=args.maxy
    minx=args.minx
    miny=args.miny
    time=args.time
    dem_directory=args.dem_directory
    working_directory=os.getcwd()+"\\work_folder"
    dpsri_directory=args.dpsri_directory
    output_file=working_directory+"\\output.xml"
    vis=args.visualize
    """
    minx = 7.261;
    maxx = 7.262;
    maxy = 43.71;
    miny = 43.70;
    time = "03 10 2015 23:55:55"
    dem_directory = "C:/Users/user/Desktop/dems"
    working_directory = "C:/Users/user/Desktop/FlashFloodApplicationfinal/work_folder"
    dpsri_directory = r"C:/Users/user/Desktop/Hassan/2015-10-03"
    output_file = working_directory + "/output.xml"
    vis = True
    """




    if (maxx<0):
        east_11="W"
        east_10="W"
        east_12="W"
    else:
        east_11="E"
        east_10="E"
        east_12="E"

    if (maxx-1<0):
        east_01="W"
        east_00="W"
        east_02="W"
    else:
        east_01="E"
        east_00="E"
        east_02="E"

    if (maxx+1<0):
        east_21="W"
        east_20="W"
        east_22="W"
    else:
        east_21="E"
        east_20="E"
        east_22="E"

    if (maxy<0):
        north_11="S"
        north_01="S"
        north_21="S"
    else:
        north_11="N"
        north_01="N"
        north_21="N"

    if (maxy-1<0):
        north_10="S"
        north_00="S"
        north_20="S"
    else:
        north_10="N"
        north_00="N"
        north_20="N"

    if (maxy+1<0):
        north_12="S"
        north_02="S"
        north_22="S"
    else:
        north_12="N"
        north_02="N"
        north_22="N"
    s_maxx=math.modf(maxx)
    s_maxy=math.modf(maxy)
    e_int_1=int(abs(s_maxx[1]))
    n_int_1=int(abs(s_maxy[1]))
    e_int_0=e_int_1-1
    e_int_2=e_int_1+1
    n_int_0=n_int_1-1
    n_int_2=n_int_1+1
    e_str_1=str(e_int_1)
    n_str_1=str(n_int_1)

    e_str_0=str(e_int_0)
    n_str_0=str(n_int_0)

    e_str_2=str(e_int_2)
    n_str_2=str(n_int_2)

    while(len(n_str_1)<2):
        n_str_1="0"+n_str_1

    while(len(e_str_1)<3):
        e_str_1="0"+e_str_1

    while(len(n_str_0)<2):
        n_str_0="0"+n_str_0

    while(len(e_str_0)<3):
        e_str_0="0"+e_str_0

    while(len(n_str_2)<2):
        n_str_2="0"+n_str_2

    while(len(e_str_2)<3):
        e_str_2="0"+e_str_2

    dem_11=dem_directory+"/"+north_11+n_str_1+east_11+e_str_1+".tif"
    dem_12=dem_directory+"/"+north_12+n_str_2+east_12+e_str_1+".tif"
    dem_10=dem_directory+"/"+north_10+n_str_0+east_10+e_str_1+".tif"
    dem_01=dem_directory+"/"+north_01+n_str_1+east_01+e_str_0+".tif"
    dem_00=dem_directory+"/"+north_00+n_str_0+east_00+e_str_0+".tif"
    dem_02=dem_directory+"/"+north_02+n_str_2+east_02+e_str_0+".tif"
    dem_21=dem_directory+"/"+north_21+n_str_1+east_21+e_str_2+".tif"
    dem_20=dem_directory+"/"+north_20+n_str_0+east_20+e_str_2+".tif"
    dem_22=dem_directory+"/"+north_22+n_str_2+east_22+e_str_2+".tif"

    range_start = 10**9
    range_end = (10**10)-1
    current=str(randint(range_start, range_end))
    files=[dem_00,dem_10,dem_20,dem_01,dem_11,dem_21,dem_02,dem_12,dem_22]
    dem_files=[]
    for file in files:
        if (os.path.exists(file)):
            dem_files.append(file)

    if not dem_files:
        print("No dem found matching the desired extent ..... exit")
        sys.exit(0)
    output_dem=working_directory+"/merged"+current+".tif"
    sources = [rasterio.open(f) for f in dem_files]
    dest, output_transform = merge_tool(sources)
    out_meta = sources[0].meta.copy()
    if (sources[0].nodatavals[0] is None):

        nodata=-32467.0
    else: nodata=sources[0].nodatavals[0]
    out_meta.update({'height': dest.shape[1],
                      'width': dest.shape[2],
                     'nodata':nodata,
                      'transform': output_transform
                     })

    with rasterio.open(output_dem, 'w', **out_meta) as d:
        d.write(dest)


    if time=="now":
        current_datetime = datetime.now()
    else:
        "h:min:sec"
        current_datetime = datetime.strptime(time,'%d %m %Y %H:%M:%S')

    extent={"minx":minx,'miny':miny,"maxx":maxx,"maxy":maxy}
    treefile = working_directory + r"\experiment\flood\run=0_numtrees=100_samples=100_depth=5_distinctions=all_stat=logrank.pkl"
    experiment_Path = working_directory+ r"\experiment\flood.exp"
    ds=DatasetGenerator(merged_dem=output_dem,now=current_datetime,past_now=2,timesteps=19,dpsri_directory=dpsri_directory,
                            extent=extent,working_directory=working_directory,cropped_name=current)
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
    t = []
    for i in range(1, 19):
        t.append(i)

    config = Config()
    config.DEBUG = True
    config['time_list'] = t
    config['exp']='flood'
    config['w_dir']=working_directory
    config['load_graphs_from_xml'] = True
    loadExperimentFile(config, filename=experiment_Path, experiment_name="flood")

    forests = pickle.load(open(treefile, 'rb'))
    if vis:
      df = forests.labelGraphs(config.graphs, config.time_list,output_file,vis)
      sort_by_time = df.sort_values('time')
      import plotly.express as px
      fig = px.scatter_mapbox(sort_by_time, lat="lat", lon="lon", hover_data=["survival_probability"],
                              color="survival_probability", animation_frame="time", animation_group="time",
                              size='danger',
                              color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10, height=500)
      fig.update_layout(mapbox_style="open-street-map")
      fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
      fig.show()

    else:
        forests.labelGraphs(config.graphs, config.time_list, output_file, vis)



if __name__ == '__main__':
    main()