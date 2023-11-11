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

def load_hudf_data(extent=None):
    """
    Load Hubble Ultra Deep Field image data for i/v/b filters from fits files. 

    Args:
        extent (list): 
         list with lower and upper index bounds for x/y axes of data [xmin, xmax, ymin, ymax]. 
         If None, uses full image
    
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

def load_hudf_z_vizier(catalog_id, zcol, mag_limit=None, limit_filter='F775W'):
    """
    Load Hubble Ultra Deep Field redshifts from Vizier catalog.

    Args:
        catalog_id (str): Vizier catalog identifier of single table to get redshifts.
        zcol (str): Table column name in Vizier catalog associated with reshifts.
        mag_limit (float): Magnitude constraint on redshift objects from Vizier catalog. If None, no constraint.
        limit_filter (str): Hubble filter to constrain magnitude.
    Returns:
        dict of full catalog, redshifts, and SkyCoords
    """
    try:
        assert isinstance(catalog_id, str)
    except: raise TypeError('catalog_id must be a string with the catalog identifier')
    try:
        assert isinstance(zcol, str)
    except: raise TypeError('zcol must be a string with the redshift column name')
    try:
        assert isinstance(limit_filter, str)
    except: raise TypeError('limit_filter must be a string with the magnitude band to constrain rows')
    if mag_limit is not None:
        try:
            assert isinstance(mag_limit, float)
        except: raise TypeError('mag_limit must be a float with the magnitude to constrain rows')


    catalog = Vizier.get_catalogs(catalog_id)
    if len(catalog) < 1:
        raise ValueError(f'No tables found with catalog id {catalog_id}')
    if len(catalog) > 1:
        raise ValueError('Please input catalog_id with a single table')
    
    catalog = catalog[0]
    # Cut rows under magnitude limit
    try:
        catalog = catalog[np.where(catalog[limit_filter] < mag_limit)]
    except KeyError: raise KeyError(f'Filter {limit_filter} is not a column in catalog')
    except TypeError:
        catalog = catalog
    # Get redshifts from catalog
    try:
        redshifts = catalog[zcol]
    except: raise KeyError(f'Name {zcol} is not a column for redshift in catalog')
    # Get coordinates from catalog
    try:
        coords = SkyCoord(catalog['RAJ2000'], catalog['DEJ2000'])
    except: raise KeyError('Catalog does not have RA/DEC columns named RAJ2000 / DEJ2000')

    args = {'catalog':catalog, 'z':redshifts, 'sky':coords}
    return args


if __name__=='__main__':
    test = load_hudf_z_vizier(catalog_id='J/AJ/150/31/table5', zcol='zph1')
    print(test)
    exit()
    ### HUDF data
    extent = [4000, 6000, 4000, 6000]

    hudf_data, hudf_header = load_hudf_data(extent)

    wcs = WCS(hudf_header[0])

    r = hudf_data[0] * 0.5
    g = hudf_data[1] * 0.3
    b = hudf_data[2] * 0.8
    t = 0.001
    r[r > t] = t
    g[g > t] = t
    b[b > t] = t


    hudf_rgb = make_lupton_rgb(r, g, b, Q=0.0001, stretch=0.001, filename='hudf.jpeg')
    
    ### Load redshifts from catalogs
    catalog_ids = {
        'photz_raf':'J/AJ/150/31/table5', 
        'specz_muse':'J/A+A/608/A2/combined'
    }
    photz_raf = load_hudf_z_vizier(catalog_ids['photz_raf'], zcol='zph1', mag_limit=25)
    specz_muse = load_hudf_z_vizier(catalog_ids['specz_muse'], zcol='zMuse', mag_limit=25)

    ### Crossmatch catalogs
    idx, sep2d, dist3d = specz_muse['sky'].match_to_catalog_sky(photz_raf['sky'])
    crossmatchz = {}
    for key, vals in photz_raf.items():
        crossmatchz[key] = vals[idx]
    print(crossmatchz)
    
    pixels_photz = wcs.world_to_pixel(photz_raf['sky'])
    pixels_specz = wcs.world_to_pixel(specz_muse['sky'])
    pixels_bothz = wcs.world_to_pixel(crossmatchz['sky'])

    # fig = plt.figure(figsize=(12,12))
    # axes = fig.add_subplot(111, projection=wcs)
    # axes.imshow(hudf_rgb, origin='lower')
    # #axes.scatter(ra, dec)
    # plt.show()
