from Raster import Raster
import richdem as rd
import whitebox
import gdal
import rasterio as rio
import georasters as gr
import sys
wbt = whitebox.WhiteboxTools()
from raster2xyz.raster2xyz import Raster2xyz
from rastertodataframe import raster_to_dataframe
rtxyz=Raster2xyz()

class DemRaster(Raster):


    def __init__(self,path,working_directory):

        Raster.__init__(self,path,working_directory)

    def crop_to_match_extent(self,cropped_name ,using_shapefile, shape_file, minx=0, miny=0, maxx=0, maxy=0):
        #cropped_path = self.working_directory + r"\gis_rasters\cropped_dem.tif"

        cropped_path = self.working_directory + r"\gis_rasters\{0}.tif".format(cropped_name)

        self.crop(self.path,cropped_path,using_shapefile,shape_file,minx,miny,maxx,maxy)
        self.cropped = cropped_path


    def  get_elevation(self):
        raster = gr.from_file(self.cropped)
        current_df = raster.to_pandas()
        current_df.drop(columns=["row", "col"], inplace=True)
        current_df.rename(columns={'value':'z', 'x': 'Lon', 'y': "Lat"}, inplace=True)
        self.elevation_df=current_df


    def get_slope(self):
        """
        :param dem_path: dem file path
        :param slope_path: slope raster path
        :param no_data: no data value
        :return: None
        """

        dem = rd.LoadGDAL(self.cropped, self.no_data)
        slope = rd.TerrainAttribute(dem, attrib='slope_degrees')
        # rd.rdShow(slope,cmap="terrain",figsize=(8,6))
        raster_path=self.working_directory+r"\gis_rasters\slope_{0}_{1}_{2}_{3}.tif".format(self.extent['minx'],
                                                                                            self.extent['maxx'],
                                                                                            self.extent['miny'],
                                                                                            self.extent['maxy'])

        rd.SaveGDAL(raster_path, slope)
        raster = gr.from_file(raster_path)
        current_df = raster.to_pandas()
        current_df.drop(columns=["row", "col"], inplace=True)
        current_df.rename(columns={'value': 'slope','x': 'Lon', 'y': "Lat"}, inplace=True)

        self.slope_df=current_df
        self.slope_raster=raster_path

    def get_plan_curvature(self):

        """
        :param plan_curvature_path: plan curvature raster path
        :return: None
        """
        dem = rd.LoadGDAL(self.cropped, self.no_data)
        plan_curvature = rd.TerrainAttribute(dem, attrib='planform_curvature')
        raster_path = self.working_directory + r"\gis_rasters\plan_curvature_{0}_{1}_{2}_{3}.tif".format(self.extent['minx'],
                                                                                            self.extent['maxx'],
                                                                                            self.extent['miny'],
                                                                                            self.extent['maxy'])

        rd.SaveGDAL(raster_path, plan_curvature)
        raster = gr.from_file(raster_path)
        current_df = raster.to_pandas()
        current_df.drop(columns=["col", "row"], inplace=True)
        current_df.rename(columns={'value': 'plan curvature','x': 'Lon', 'y': "Lat"}, inplace=True)
        self.plan_curvature_df=current_df

    def get_aspect(self):

        """
        :param aspect_path: aspect raster path
        :return: None
        """

        dem = rd.LoadGDAL(self.cropped, self.no_data)
        aspect = rd.TerrainAttribute(dem, attrib='aspect')
        raster_path = self.working_directory + r"\gis_rasters\aspect_{0}_{1}_{2}_{3}.tif".format(
            self.extent['minx'],
            self.extent['maxx'],
            self.extent['miny'],
            self.extent['maxy'])
        rd.SaveGDAL(raster_path, aspect)
        raster = gr.from_file(raster_path)
        current_df = raster.to_pandas()
        current_df.drop(columns=["row", "col"], inplace=True)
        current_df.rename(columns={'value': 'aspect','x': 'Lon', 'y': "Lat"}, inplace=True)
        self.aspect_df=current_df

    # inner parameter
    def get_dem_correction(self):

        """
        :param corrected_dem_path: corrected dem raster path
        :return: None
        """
        corrected_dem_path=self.working_directory + r"\gis_rasters\dem_corrected_{0}_{1}_{2}_{3}.tif".format(
            self.extent['minx'],
            self.extent['maxx'],
            self.extent['miny'],
            self.extent['maxy'])

        dem = rd.LoadGDAL(self.cropped, no_data=self.no_data)
        correcteddem = rd.FillDepressions(dem, epsilon=True, in_place=False, topology="D8")

        rd.SaveGDAL(corrected_dem_path, correcteddem)

        self.corrected_path=corrected_dem_path

    # inner parameter
    def get_upslope_contributing_area(self):

        """
        :param dinf_path: flow accumulation raster using dinf method
        :param sca_path: specific contributing area raster path
        :return: None
        """
        dinf_path = self.working_directory + r"\gis_rasters\dinf_{0}_{1}_{2}_{3}.tif".format(
            self.extent['minx'],
            self.extent['maxx'],
            self.extent['miny'],
            self.extent['maxy'])
        raster_path = self.working_directory + r"\gis_rasters\sca_{0}_{1}_{2}_{3}.tif".format(
            self.extent['minx'],
            self.extent['maxx'],
            self.extent['miny'],
            self.extent['maxy'])
        format1 = "GTiff"
        driver = gdal.GetDriverByName(format1)
        rio_dataset = rio.open(self.cropped)
        flow_length = rio_dataset.res[0]
        correcteddem = rd.LoadGDAL(self.corrected_path, self.no_data)
        accum_dinf = rd.FlowAccumulation(correcteddem, method='Dinf')
        rd.SaveGDAL(dinf_path, accum_dinf)
        accum_pr = gdal.Open(dinf_path)
        geotransform = accum_pr.GetGeoTransform()
        geoproj = accum_pr.GetProjection()
        xsize = accum_pr.RasterXSize
        ysize = accum_pr.RasterYSize
        array = accum_pr.ReadAsArray()
        array = array.astype('f')
        dst_datatype = gdal.GDT_Float32
        cal_array = array / 30
        dst_ds = driver.Create(raster_path, xsize, ysize, 1, dst_datatype)
        dst_ds.GetRasterBand(1).WriteArray(cal_array)
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(geoproj)
        dst_ds = None
        self.sca_path=raster_path

    def get_stream_power_index(self):

        """
        :param spi_path: stream power index raster path
        :return: None
        """
        raster_path = self.working_directory + r"\gis_rasters\spi_{0}_{1}_{2}_{3}.tif".format(
            self.extent['minx'],
            self.extent['maxx'],
            self.extent['miny'],
            self.extent['maxy'])

        wbt.relative_stream_power_index(self.sca_path,self.slope_raster,raster_path, exponent=1.0)
        raster = gr.from_file(raster_path)
        current_df = raster.to_pandas()
        current_df.drop(columns=["row", "col"], inplace=True)
        current_df.rename(columns={'value': 'stream power index','x': 'Lon', 'y': "Lat"}, inplace=True)
        self.spi_df=current_df

    def get_upslope_flow_path_length(self):

        """
        :param avg_upslope_flow_path_length: average upslope flow path length raster path
        :return: None
        """
        raster_path = self.working_directory + r"\gis_rasters\flow_length_{0}_{1}_{2}_{3}.tif".format(
            self.extent['minx'],
            self.extent['maxx'],
            self.extent['miny'],
            self.extent['maxy'])

        wbt.average_upslope_flowpath_length(self.corrected_path, raster_path)
        raster = gr.from_file(raster_path)
        current_df = raster.to_pandas()
        current_df.drop(columns=["row", "col"], inplace=True)
        current_df.rename(columns={'value': 'flow length','x': 'Lon', 'y': "Lat"}, inplace=True)
        self.flow_length_df=current_df

    def get_wetness_index(self):

        """
        :param sca_path: specific contributing area raster path
        :param slope_path: slope raster path
        :param wetness_index_path: topographic wetness index raster path
        :return: None
        """
        import pandas as pd
        import numpy as np
        raster_path = self.working_directory + r"\gis_rasters\wetness_index_{0}_{1}_{2}_{3}.tif".format(
            self.extent['minx'],
            self.extent['maxx'],
            self.extent['miny'],
            self.extent['maxy'])

        wbt.wetness_index(self.sca_path, self.slope_raster, raster_path)
        raster = gr.from_file(raster_path)
        current_df = raster.to_pandas()
        current_df.drop(columns=["row", "col"], inplace=True)
        current_df.rename(columns={'value': 'wetness index','x': 'Lon', 'y': "Lat"}, inplace=True)
        self.wi_df=current_df

    def get_distance_to_stream(self):
        """
       
        :param accum_d8_path: flow accumulation using D8 raster path
        :param streams_path: streams raster path 
        :param filled_flat_resolved_dem_path: filled flat resolved dem raster path
        :param downslope_distance_to_streams: downslope distance to stream raster path
        :return: None
        """

        accum_d8_path=self.working_directory + r"\gis_rasters\accum_d8_{0}_{1}_{2}_{3}.tif".format(
            self.extent['minx'],
            self.extent['maxx'],
            self.extent['miny'],
            self.extent['maxy'])
        streams_path= self.working_directory + r"\gis_rasters\streams_{0}_{1}_{2}_{3}.tif".format(
            self.extent['minx'],
            self.extent['maxx'],
            self.extent['miny'],
            self.extent['maxy'])
        filled_flat_resolved_dem_path=self.working_directory + r"\gis_rasters\flat_resolved_{0}_{1}_{2}_{3}.tif".format(
            self.extent['minx'],
            self.extent['maxx'],
            self.extent['miny'],
            self.extent['maxy'])
        raster_path=self.working_directory + r"\gis_rasters\distance_to_streams_{0}_{1}_{2}_{3}.tif".format(
            self.extent['minx'],
            self.extent['maxx'],
            self.extent['miny'],
            self.extent['maxy'])
        dem = rd.LoadGDAL(self.cropped, self.no_data)
        correcteddem = rd.FillDepressions(dem, epsilon=True, in_place=False, topology="D8")
        accum_d8 = rd.FlowAccumulation(correcteddem, method='D8')
        rd.SaveGDAL(accum_d8_path, accum_d8)
        wbt.extract_streams(accum_d8_path, streams_path, threshold=25, zero_background=True)
        flat_resolved = rd.ResolveFlats(dem, in_place=False)
        filled_flat_resolved = rd.FillDepressions(flat_resolved, epsilon=True, in_place=False, topology="D8")
        rd.SaveGDAL(filled_flat_resolved_dem_path, filled_flat_resolved)
        wbt.downslope_distance_to_stream(filled_flat_resolved_dem_path, streams_path, raster_path)
        raster = gr.from_file(raster_path)
        current_df = raster.to_pandas()
        current_df.drop(columns=["row", "col"], inplace=True)
        current_df.rename(columns={'value': 'distance to stream','x': 'Lon', 'y': "Lat"}, inplace=True)
        self.distance_to_stream_df=current_df

