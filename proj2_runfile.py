""" ASTR 142 Project 2 module containing plotter class for Hubble Ultra Deep Field. """
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

class HUDF_Plotter():
    """
    Plotter for the Hubble Ultra Deep Field (HUDF) using RGB image and redshift catalogs.

    - Can plot objects from redshift catalogs and redshift distributions.
    - Can plot smaller square subregion views in multi-panel configuration.

    Parameters
    ------------
    data_path : str 
        Path to RGB HUDF data files (`h_udf_wfc_{i/v/b}_drz_img.fits`)
    rgb_scale : tuple(float float, float)
        Scaling values for RGB data before converting to RGB image in `make_lupton_rgb()`.
    rgb_max : float 
        Maximum color value for RGB values before converting to RGB image.
    debug : bool 
        If True, load part of HUDF data array to speed up.

    Attributes
    ----------
        data (list): List of RGB data arrays from HUDF fits files.
        header (list): List of headers from HUDF fits files.
        wcs (astropy.WCS): WCS object associated with the RGB data headers.
        rgb (ndarray): A NxNx3 RGB integer array to be plotted with `imshow`.
        z_catalogs (dict): Dictionary of `HUDF_z_Catalog` objects to be plotted.
        z_colors (dict): Dictionary of matplotlib colors to plot catalogs.
        inset_coords (SkyCoord): SkyCoord of all RA/DEC positions of the centers of square subregions to plot. Max 9 insets.
        inset_sizes (list): List of sizes of square subregions associated with `inset_coords`, in degrees. Max 9 insets.
    """
    def __init__(self, data_path='data', rgb_scale=(0.6, 0.3, 0.9), rgb_max=0.001, debug=False):
        self.data_path = data_path
        self.rgb_scale = rgb_scale
        self.rgb_max = rgb_max
        self.data, self.header = self.load_hudf_data(debug)
        self.wcs = WCS(self.header[0])
        logging.info('Converting to RGB image...')
        self.rgb = make_lupton_rgb(self.data[0], self.data[1], self.data[2], Q=0.0001, stretch=0.001)
        self.z_catalogs = {}
        self.z_colors = {}
        self.inset_coords = None

    def load_hudf_data(self, debug):
        """
        Load Hubble Ultra Deep Field image data for i/v/b filters from fits files.

        Returns:
            Two lists of length 3 containing the RGB image data and the header of each fits file.
        """
        if debug:
            extent = [5000, 5500, 5000, 5500]
        else: extent = None
        hudf_data = []
        hudf_header = []
        logging.info('Loading Hubble UDF fits files...')
        for i, c in enumerate(['i', 'v', 'b']):
            logging.info(f'  Loading {c} image')
            with fits.open(f'{self.data_path}/h_udf_wfc_{c}_drz_img.fits') as hdul:
                hudf_header.append(hdul[0].header)
                if extent is not None:
                    hudf_data.append(hdul[0].data[extent[0]:extent[1], extent[2]:extent[3]])
                    continue
                hudf_data.append(hdul[0].data * self.rgb_scale[i])
                hudf_data[i][hudf_data[i] > self.rgb_max] = self.rgb_max
        logging.info('Finished loading.')
        return hudf_data, hudf_header

    def add_z_catalog(self, catalog_id, zcol, mag_constraint=None, limit_filter='F775W', catalog_key=None, color=None):
        """
        Add magnitude-constrained redshift catalog object to be plotted.

        Parameters
        ------------
            catalog_id (str): The catalog ID of redshift catalog to be queried by Vizier. Catalog must be a single table.
            zcol (str): The column name for the desired redshift column of the catalog.
            mag_constraint (float): The magnitude constraint in the filter specified in `limit_filter`. If `None`, no magnitude constraint.
            limit_filter (str): Filter in which the magnitudes are constrained by `mag_constraint` (Default to F775W filter of Hubble).
            catalog_key (str): The key associated with the `HUDF_z_Catalog` object in `z_catalogs`. If `None`, use `catalog_id`.
            color (str): The matplotlib color to plot redshift objects in. If None, will plot default matplotlib colors.
        Returns:
            HUDF_z_Catalog object with associated catalog.
        """
        catalog = HUDF_z_Catalog(catalog_id, zcol, mag_constraint, limit_filter)
        if catalog_key is None:
            catalog_key = catalog_id
        self.z_catalogs[catalog_key] = catalog
        self.z_colors[catalog_key] = color
        return catalog
    
    def remove_z_catalog(self, catalog_key):
        """
        Remove catalog object from `z_catalogs`.

        Parameters
        ------------
            catalog_key (str): The catalog key associated with `HUDF_z_Catalog` object in `z_catalogs`.
        Returns:
            The removed `HUDF_z_Catalog` object. If `catalog_key` not found, return None.
        """
        if catalog_key in self.z_catalogs.keys():
            removed_cat = self.z_catalogs.pop(catalog_key)
            self.z_colors.pop(catalog_key)
            return removed_cat
        else:
            logging.warning(f'Key {catalog_key} not found in `z_catalogs`.')
            return None
        

    def cross_match_catalogs(self, catalog_key1, catalog_key2, new_key=None, new_color=None):
        """
        Cross-match redshift catalogs.

        Parameters
        ------------
            catalog_key1 (str): The key in `z_catalogs` associated with the first catalog to be cross-matched.
            catalog_key2 (str): The key in `z_catalogs` associated with the second catalog to be cross-matched.
            new_key (str): The new key in  `z_catalogs` for the cross-matched catalog object.
            new_color (str): The matplotlib color to plot the cross-matched catalog. If None, will plot default matplotlib colors.
        Returns:
            Key of cross-matched catalog `HUDF_z_Catalog` object added to `z_catalogs`.
            Cross-matched catalog object will have same attributes as the second catalog, except with cross-matched catalog.
        """
        try:
            cat1 = self.z_catalogs[catalog_key1]
            cat2 = self.z_catalogs[catalog_key2]
        except: raise KeyError('`catalog_key1` and `catalog_key2` must be keys in `z_catalogs`')

        idx, sep2d, dist3d = cat1.coords.match_to_catalog_sky(cat2.coords)
        idx = list(set(idx)) # Remove duplicate indices
        new_catalog = HUDF_z_Catalog(cat2.catalog_id, cat2.zcol, cat2.mag_constraint, cat2.limit_filter)
        new_catalog.catalog = cat2.catalog[idx]
        new_catalog.zs = cat2.zs[idx]
        new_catalog.coords = cat2.coords[idx]
        if new_key is None:
            new_key = catalog_key1 + '_' + catalog_key2
            logging.info(f'Key for cross-matched catalog not specified. Defaulting to {new_key}')
        self.z_catalogs[new_key] = new_catalog
        self.z_colors[new_key] = new_color

    def plot(self, plot_catalogs=True, ax=None):
        """
        Plot single panel with Hubble Ultra Deep Field image and (optionally) objects from redshift catalogs and redshift distribution.

        Parameters
        ----------
        plot_catalogs : bool or list
            If True, plot all catalogs in `z_catalogs`. If False, plot no catalogs. If list of keys provided, plot only catalogs associated with keys in `z_catalogs`.
        ax : plt.Axes
            The Axes to plot image on. The Axes object must have the correct WCS projection. 
            If None, create new figure and axes and return both.
        
        Returns
        -------
        ax : plt.Axes
            The Axes on which the image is plotted. 
        fig : plt.Figure
            The new Figure that includes ax. If no new Figure created, return None.
        """
        ### Get catalogs to plot
        plot_cats = {}
        if isinstance(plot_catalogs, list):
            for key in plot_catalogs:
                if key not in self.z_catalogs.keys():
                    logging.warning(f'Key {key} not found in `z_catalogs`. Skipping.')
                    continue
                plot_cats[key] = self.z_catalogs[key]
        elif plot_catalogs:
                plot_cats = self.z_catalogs
        
        ### Plot
        fig = None
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=self.wcs)
        im = ax.imshow(self.rgb, origin='lower')
        for key, cat in plot_cats.items():
            coords = cat.coords
            pixel_coords = self.wcs.world_to_pixel(coords)
            ax.scatter(pixel_coords[0], pixel_coords[1], s=1, c=self.z_colors[key], label=key)
        ax.legend()
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('DEC (deg)')
        return fig, ax
    
    def plot_z_dist(self, plot_catalogs=None, bins=None, ax=None):
        """
        Plot redshift distributions of catalogs in `z_catalogs`.

        Parameters
        ------------
            plot_catalogs (list): List of keys in `z_catalogs` to plot.
            bins (int or list or None): Same `bins` parameter as in `plt.hist`. If None, use default bins for redshifts.
        ax : plt.Axes
            The Axes to plot image on. The Axes object must have the correct WCS projection. 
            If None, create new figure and axes and return both.
        
        Return:
            Figure and Axes object used to plot.
        """
        ### Get catalogs to plot
        hist_cats = {}
        if plot_catalogs is None:
            hist_cats = self.z_catalogs
        elif isinstance(plot_catalogs, list):
            for key in plot_catalogs:
                if key not in self.z_catalogs.keys():
                    logging.warning(f'Key {key} not found in `z_catalogs`. Skipping.')
                    continue
                hist_cats[key] = self.z_catalogs[key]

        ### Set up same bins for all hists
        if bins is None:
            bins = np.arange(0, 7, 0.1)

        ### Plot
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        for key, cat in hist_cats.items():
            zs = cat.zs
            ax.hist(zs, color=self.z_colors[key], label=key, alpha=0.5, bins=bins)
        ax.legend()
        ax.set_xlabel('Redshift z')
        ax.set_ylabel('Count')

    def plot_multipanel(self, figsize=(12, 24)):
        fig = plt.figure(figsize=figsize)


class HUDF_z_Catalog():
    """
    Catalog of redshifts (z) from Vizier catalog of Hubble Ultra Deep Field objects.

    Parameters
    ------------
        catalog_id (str): The catalog ID of redshift catalog to be queried by Vizier. Catalog must be a single table.
        zcol (str): The column name for the desired redshift column of the catalog.
        mag_constraint (float): The magnitude constraint in the filter specified in `limit_filter`. If `None`, no magnitude constraint.
        limit_filter (str): Filter in which the magnitudes are constrained by `mag_constraint` (Default to F775W filter of Hubble).

    Attributes:
        catalog (astropy.Table): Catalog table result from Vizier query, magnitude constrained by `mag_constraint` and `limit_filter`.
        zs (astropy.MaskedColumn): Redshift column from catalog in the `zcol` column.
        coords (SkyCoord): SkyCoord of RA/DEC positions of objects from catalog.
    """
    def __init__(self, catalog_id, zcol, mag_constraint=None, limit_filter='F775W'):
        try:
            assert isinstance(catalog_id, str)
        except: raise TypeError('`catalog_id` must be a string with the catalog identifier.')
        try:
            assert isinstance(zcol, str)
        except: raise TypeError('`zcol` must be a string with the redshift column name.')
        try:
            assert isinstance(limit_filter, str)
        except: raise TypeError('`limit_filter` must be a string with the magnitude filter to constrain rows.')
        if mag_constraint is not None:
            try:
                mag_constraint = float(mag_constraint)
            except: raise TypeError('`mag_constraint` must be a float with the magnitude to constrain rows.')

        self.catalog_id = catalog_id
        self.zcol = zcol
        self.mag_constraint = mag_constraint
        self.limit_filter = limit_filter
        self.catalog, self.zs, self.coords = self.load_hudf_z_vizier()

    def load_hudf_z_vizier(self):
        """ Load catalog and constrains rows with `mag_constraint` and `limit_filter`. """

        catalog = Vizier.get_catalogs(self.catalog_id)
        if len(catalog) < 1:
            raise ValueError(f'No tables found with `catalog_id` {self.catalog_id}.')
        if len(catalog) > 1:
            raise ValueError('Catalog associated with `catalog_id` must have one table.')

        catalog = catalog[0]
        # Cut rows under magnitude limit
        if self.mag_constraint is None:
            catalog = catalog
        else:
            try:
                catalog = catalog[np.where(catalog[self.limit_filter] < self.mag_constraint)]
            except: raise KeyError(f'Filter {self.limit_filter} is not a column in catalog')

        # Get redshifts from catalog
        try:
            redshifts = catalog[self.zcol]
        except: raise KeyError(f'Name {self.zcol} is not a column for redshift in catalog')

        # Get coordinates from catalog
        try:
            coords = SkyCoord(catalog['RAJ2000'], catalog['DEJ2000'])
        except: raise KeyError('Catalog does not have RA/DEC columns named `RAJ2000` / `DEJ2000`')

        return catalog, redshifts, coords



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

def load_hudf_z_vizier(catalog_id, zcol, mag_limit=24, limit_filter='F775W'):
    """ Load catalog and constrains rows with `mag_constraint` and `limit_filter`. """

    catalog = Vizier.get_catalogs(catalog_id)
    if len(catalog) < 1:
        raise ValueError(f'No tables found with `catalog_id` {catalog_id}.')
    if len(catalog) > 1:
        raise ValueError('Catalog associated with `catalog_id` must have one table.')

    catalog = catalog[0]
    # Cut rows under magnitude limit
    if mag_limit is None:
        catalog = catalog
    else:
        try:
            catalog = catalog[np.where(catalog[limit_filter] < mag_limit)]
        except: raise KeyError(f'Filter {limit_filter} is not a column in catalog')

    # Get redshifts from catalog
    try:
        redshifts = catalog[zcol]
    except: raise KeyError(f'Name {zcol} is not a column for redshift in catalog')

    # Get coordinates from catalog
    try:
        coords = SkyCoord(catalog['RAJ2000'], catalog['DEJ2000'])
    except: raise KeyError('Catalog does not have RA/DEC columns named `RAJ2000` / `DEJ2000`')

    args = {'cat':catalog, 'z':redshifts, 'sky':coords}
    return args


if __name__=='__main__':
    ### Load HUDF data
    extent = [4000, 6000, 4000, 6000]

    hudf_data, hudf_header = load_hudf_data()

    wcs = WCS(hudf_header[0])

    print('Configuring colors...')
    r = hudf_data[0] * 0.5
    g = hudf_data[1] * 0.3
    b = hudf_data[2] * 0.8
    t = 0.001
    r[r > t] = t
    g[g > t] = t
    b[b > t] = t
    print('Done configuring.')

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

    pixels_photz = wcs.world_to_pixel(photz_raf['sky'])
    pixels_specz = wcs.world_to_pixel(specz_muse['sky'])
    pixels_bothz = wcs.world_to_pixel(crossmatchz['sky'])

    fig = plt.figure(figsize=(12,12))
    axes = fig.add_subplot(111, projection=wcs)
    axes.imshow(hudf_rgb, origin='lower')
    axes.scatter(pixels_photz[0], pixels_photz[1], s=3, c='y')
    axes.scatter(pixels_specz[0], pixels_specz[1], s=3, c='m')
    axes.scatter(pixels_bothz[0], pixels_bothz[1], s=3, c='c')

    plt.show()