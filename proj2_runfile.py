import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm
import sys
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = 10000
from astropy.coordinates import Angle
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def load_HUDF_data(extent=None):
    """
    Load Hubble Ultra Deep Field image data for i/v/b filters from fits files. 

    Args:
        extent: 
         list with lower and upper index bounds for x/y axes of data. [xmin, xmax, ymin, ymax]
    
    Returns:
        Two lists of length 3 containing the RGB image data and the header of each fits file.
    
    Raises:
        TypeError: if extent is not a list
        ValueError: if extent is not a list of length 4
    """
    if extent is not None:
        try:
            assert isinstance(extent, list)
        except:
            raise TypeError('extent must be a list of length 4')
        try:
            assert len(extent) == 4
        except:
            raise ValueError('extent must be a list of length 4')
        
    hudf_data = []
    hudf_header = []
    logging.info('Loading Hubble UDF fits files...')
    for i, c in enumerate(['i', 'v', 'b']):
        with fits.open(f'data/h_udf_wfc_{c}_drz_img.fits') as hdul:
            if extent is not None:
                hudf_data.append(hdul[0].data[extent[0]:extent[1], extent[2]:extent[3]])
            else: hudf_data.append(hdul[0].data)
            hudf_header.append(hdul[0].header)
    logging.info('Finished loading.')
    return hudf_data, hudf_header

if __name__=='__main__':
    extent = [4000, 6000, 4000, 6000]

    hudf_data, hudf_header = load_HUDF_data(extent)

    wcs = WCS(hudf_header[0])

    r = hudf_data[0] * 0.5
    g = hudf_data[1] * 0.3
    b = hudf_data[2] * 0.8

    hudf_rgb = make_lupton_rgb(r, g, b, Q=0.0001, stretch=0.004, filename='hudf.jpeg')


    Vizier.ROW_LIMIT = 10000
    catalog_name = 'J/AJ/150/31'
    catalog = Vizier.get_catalogs(catalog_name)
    photo_z_catalog = catalog[0]
    print(photo_z_catalog)
    photo_z1 = photo_z_catalog['zph1']
    photo_z2 = photo_z_catalog['zph2']
    ra = photo_z_catalog['RAJ2000']
    dec = photo_z_catalog['DEJ2000']

    fig = plt.figure(figsize=(12,12))
    axes = fig.add_subplot(111, projection=wcs)
    axes.imshow(hudf_rgb, origin='lower')
    axes.scatter(ra, dec)
    plt.show()
