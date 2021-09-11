from Raster import Raster
import whitebox
import gdal
import subprocess
wbt = whitebox.WhiteboxTools()
from gdalconst import GA_ReadOnly
from osgeo import gdal

class DpsriRaster(Raster):

    def __init__(self,path,working_directory):

        Raster.__init__(self,path,working_directory)

        lst = self.path.split('\\')
        self.filename = lst[-1]



    def reproject(self):
        reprojected_path = self.working_directory + r"\dpsri_rasters\reprojected" + r"\\" + self.filename
        data = gdal.Open(self.path, GA_ReadOnly)
        gdal.Warp(reprojected_path, data, dstSRS='EPSG:4326')
        self.reprojected=reprojected_path


    def crop_to_match_extent(self,using_shapefile,shape_file,minx=0,miny=0,maxx=0,maxy=0):
        """
        
        :param cropped_path: 
        :param using_shapefile: 
        :param shape_file: 
        :param minx: input coo provided by the user
        :param miny: input coo provided by the user
        :param maxx: input coo provided by the user
        :param maxy: input coo provided by the user
        :return: 
        """

        cropped_path=self.working_directory+r"\dpsri_rasters\cropped_to_extent"+r"\\"+self.filename
        self.crop(self.upsampled,cropped_path,using_shapefile,shape_file,minx,miny,maxx,maxy)
        self.cropped_to_extent=cropped_path


    def crop_to_upsample(self,using_shapefile,shape_file,minx=0,miny=0,maxx=0,maxy=0):
        """
        
        :param cropped_path: 
        :param using_shapefile: 
        :param shape_file: 
        :param minx: coo of the original dem downloaded from nasa
        :param miny: coo of the original dem downloaded from nasa
        :param maxx: coo of the original dem downloaded from nasa
        :param maxy: coo of the original dem downloaded from nasa
        :return: 
        """
        cropped_path = self.working_directory + r"\dpsri_rasters\cropped_for_upsampling"  +r'\\'+ self.filename
        self.crop(self.reprojected,cropped_path,using_shapefile,shape_file,minx,miny,maxx,maxy)
        self.cropped_to_upsample=cropped_path


    def upsample(self,width,height):
        """
        
        :param width: width of the original dem downloaded from nasa
        :param height: height of the original dem downloaded from nasa
        :param original_path: cropped to usample dpsri raster
        :param upsampled_path: upsampled dpsri raster
        :return: 
        """
        upsampled_path = self.working_directory + r"\dpsri_rasters\upsampled" + "\\" + self.filename
        subprocess.call(
            'gdal_translate -of GTiff -outsize {0} {1} -r average {2} {3}'.format(width,height,self.cropped_to_upsample,
                                                                                  upsampled_path))
        self.upsampled=upsampled_path
