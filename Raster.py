
import  rasterio as rio
import gdal
import subprocess
from gdalconst import *
import geopandas as gpd
class Raster(object):

    def __init__(self,path,working_directory):
        self.path=path
        self.no_data=self.get_no_data()
        self.original_extent=self.get_extent(self.path)
        self.get_dimensions(self.path)
        self.working_directory=working_directory


    def get_dimensions(self,path):
        with rio.open(path)as dem_rio:
            self.height = dem_rio.height
            self.width = dem_rio.width


    def get_no_data(self):
        
        dem_gdal = gdal.Open(self.path)
        band = dem_gdal.GetRasterBand(1)
        no_data = band.GetNoDataValue()
        return no_data


    def get_extent(self,path):
        data = gdal.Open(self.path, GA_ReadOnly)
        geoTransform = data.GetGeoTransform()
        minx = geoTransform[0]
        maxy = geoTransform[3]
        maxx = minx + geoTransform[1] * data.RasterXSize
        miny = maxy + geoTransform[5] * data.RasterYSize
        extent={"maxx":maxx, "maxy":maxy, "minx":minx, "miny":miny}
        return extent

    def crop(self,original_path,cropped_path, using_shapefile, shape_file, minx=0, miny=0, maxx=0, maxy=0):
        if using_shapefile:
            cropping_vector = gpd.read_file(shape_file)
            minx = cropping_vector.bounds.loc[0, 'minx']
            miny = cropping_vector.bounds.loc[0, 'miny']
            maxx = cropping_vector.bounds.loc[0, 'maxx']
            maxy = cropping_vector.bounds.loc[0, 'maxy']
        subprocess.call('gdal_translate -projwin ' + ' '.join(
            [str(x) for x in [minx, maxy, maxx, miny]]) + ' -of GTiff {0} {1}'.format(
            original_path, cropped_path), shell=True)
        extent = {"maxx": maxx, "maxy": maxy, "minx": minx, "miny": miny}
        self.extent = extent



