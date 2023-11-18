""" ASTR 142 Project 2 module containing plotter class for Hubble Ultra Deep Field. """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from math import ceil
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
from astroquery.vizier import Vizier
import os
Vizier.ROW_LIMIT = 10000
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

class HUDF_Plotter():
    """
    Plotter for the Hubble Ultra Deep Field (HUDF) using RGB image and redshift catalogs.

    - Can plot objects from redshift catalogs and redshift distributions.
    - Can plot smaller square subregion views in multi-panel configuration.

    Attributes
    ----------
    data_path : str 
        Path to RGB HUDF data files (`h_udf_wfc_{i/v/b}_drz_img.fits`)
    rgb_scale : (float float, float)
        Scaling values for RGB data before converting to RGB image in `make_lupton_rgb()`.
    rgb_max : float 
        Maximum color value for RGB values before converting to RGB image.
    data : list
        List of RGB data arrays from HUDF fits files.
    header : list
        List of headers from HUDF fits files.
    wcs : astropy.WCS
        WCS object associated with the RGB data headers.
    rgb : ndarray
        A NxNx3 RGB integer array to be plotted with `imshow`.
    z_catalogs : dict
        Dictionary of `HUDF_z_Catalog` objects to be plotted.
    z_colors : dict
        Dictionary of matplotlib colors to plot catalogs.
    inset_coords : SkyCoord
        SkyCoord of all RA/DEC positions of the centers of square subregions to plot.
    inset_sizes : list
        List of pixel sizes of square subregions associated with `inset_coords`.

    Methods
    -------
    __init__(data_path='data', rgb_scale=(0.6, 0.3, 0.9), rgb_max=0.001, debug=False)
        Initialize plotter by loading RGB data.
    load_hudf_data(debug):
        Load Hubble Ultra Deep Field image data for i/v/b filters from fits files.
    add_z_catalog(self, catalog_id, zcol, mag_constraint=None, limit_filter='F775W', catalog_key=None, color=None)
        Add magnitude-constrained redshift catalog object to be plotted.
    remove_z_catalog(self, catalog_key)
        Remove catalog object from `z_catalogs`.
    cross_match_catalogs(self, catalog_key1, catalog_key2, new_key=None, new_color=None)
        Cross-match redshift catalogs.
    get_plot_catalogs(self, plot_catalogs=True)
        Helper function to get catalogs in `z_catalogs` by keys in `plot_catalogs`.
    plot(self, extent=None, plot_catalogs=True, ax=None)
        Plot single panel with Hubble Ultra Deep Field image and (optionally) objects from 
        redshift catalogs and redshift distribution.
    plot_z_hist(self, plot_catalogs=True, bins=None, ax=None)
        Plot redshift distributions of catalogs in `z_catalogs` as histogram.
    add_inset(self, coords, sizes)
        Add inset views of square subregions centered at given coordinates to `inset_coords`.
        If coords is outside image, ignore coords.
    plot_multipanel(self, plot_catalogs=True, bins=None, figsize=(20,10))
        Plot multi-panel configuration, including HUDF image and overlaid redshift objects, redshift distribution,
        and inset views of subregions.
    """

    def __init__(self, data_path='data/', rgb_scale=(0.6, 0.3, 0.9), rgb_max=0.001, debug=False):
        """
        Initialize plotter by loading RGB data.

        Parameters
        ------------
        data_path : str 
            Path to RGB HUDF data files (`h_udf_wfc_{i/v/b}_drz_img.fits`)
        rgb_scale : (float float, float)
            Scaling values for RGB data before converting to RGB image in `make_lupton_rgb()`.
        rgb_max : float 
            Maximum color value for RGB values before converting to RGB image.
        debug : bool 
            If True, load part of HUDF data array to speed up.
        """
        if not isinstance(data_path, str):
            raise TypeError('`data_path` must be a string to the directory including the fits files.')
        if not isinstance(rgb_scale, tuple) or len(rgb_scale) != 3:
            raise TypeError('`rgb_scale` must a tuple of length 3 including scale factors for RGB images.')
        try: float(rgb_max)
        except: raise TypeError('`rgb_max` must be a float with max value for RGB images.')
        self.data_path = data_path
        self.rgb_scale = rgb_scale
        self.rgb_max = rgb_max
        try:
            self.data, self.header = self.load_hudf_data(debug)
        except FileNotFoundError: 
            raise FileNotFoundError(f'Cannot find HUDF fits files in `{self.data_path}/`. Fits files should be named `h_udf_wfc_[c]_drz_img.fits` where [c] is i,v,b.')
        self.wcs = WCS(self.header[0])
        logging.info('Converting to RGB image...')
        self.rgb = make_lupton_rgb(self.data[0], self.data[1], self.data[2], Q=0.0001, stretch=0.001)
        self.z_catalogs = {}
        self.z_colors = {}
        self.inset_coords = None
        self.inset_pixels = np.array([[],[]])
        self.inset_sizes = np.array([])

    def load_hudf_data(self, debug):
        """
        Load Hubble Ultra Deep Field image data for i/v/b filters from fits files.

        Parameters
        ------------
        debug : bool 
            If True, load part of HUDF data array to speed up.
        
        Returns
        -------
        hudf_data : list
            List of length 3 containing the RGB image data.
        hudf_header : list
            List of length 3 containing the header of each fits file.
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
        catalog_id : str
            The catalog ID of redshift catalog to be queried by Vizier. Catalog must be a single table.
        zcol : str
            The column name for the desired redshift column of the catalog.
        mag_constraint : float
            The magnitude constraint in the filter specified in `limit_filter`. If `None`, no magnitude constraint.
        limit_filter : str
            Filter in which the magnitudes are constrained by `mag_constraint` (Default to F775W filter of Hubble).
        catalog_key : str
            The key associated with the `HUDF_z_Catalog` object in `z_catalogs`. If `None`, use `catalog_id`.
        color : str
            The matplotlib color to plot redshift objects in. If None, will plot default matplotlib colors.
        
        Returns
        -------
        catalog : `HUDF_z_Catalog`
            `HUDF_z_Catalog` object with associated catalog.
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
        catalog_key : str
            The catalog key associated with `HUDF_z_Catalog` object in `z_catalogs`.
        
        Returns
        -------
        removed_cat : HUDF_z_Catalog or None
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
        catalog_key1 : str
            The key in `z_catalogs` associated with the first catalog to be cross-matched.
        catalog_key2 : str
            The key in `z_catalogs` associated with the second catalog to be cross-matched.
        new_key : str
            The new key in `z_catalogs` for the cross-matched catalog object.
        new_color : str
            The matplotlib color to plot the cross-matched catalog. 
            If None, will plot default matplotlib colors.
        
        Returns
        -------
        new_catalog : `HUDF_z_Catalog`
            `HUDF_z_Catalog` object associated with cross-matched catalog, which will have the same
            attributes as the second catalog, but with reduced catalog list.
        new_key : str
            Key of cross-matched catalog `HUDF_z_Catalog` object added to `z_catalogs`.
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
        return new_catalog, new_key

    def get_plot_catalogs(self, plot_catalogs=True):
        """ 
        Helper function to get catalogs in `z_catalogs` by keys in `plot_catalogs`.
        
        Parameters
        ----------
        plot_catalogs : list(str) or bool
            List of catalog keys in `z_catalogs` specified for plotting. 
            If True, return `z_catalogs`. If False, return empty dict.
        
        Returns
        -------
        plot_dict : dict
            Dictionary of `HUDF_z_Catalog` objects to plot.
        
        """
        plot_dict = {}
        if isinstance(plot_catalogs, list):
            for key in plot_catalogs:
                if key not in self.z_catalogs.keys():
                    logging.warning(f'Key {key} not found in `z_catalogs`. Skipping.')
                    continue
                plot_dict[key] = self.z_catalogs[key]
        elif plot_catalogs:
            plot_dict = self.z_catalogs
        return plot_dict

    def plot(self, extent=None, plot_catalogs=True, ax=None):
        """
        Plot single panel with Hubble Ultra Deep Field image and (optionally) objects from 
        redshift catalogs and redshift distribution.

        Parameters
        ----------
        extent : list or None
            List of [xmin, xmax, ymin, ymax] pixel coordinates to plot. If None, plot all. 
            If extent is set, `plot` will not plot catalogs.
        plot_catalogs : list(str) or bool
            List of catalog keys in `z_catalogs` specified for plotting. 
            If True, return `z_catalogs`. If False, return empty dict.
        ax : plt.Axes
            The Axes to plot image on. The Axes object must have the correct WCS projection. 
            If None, create new figure and axes and return both.
        
        Returns
        -------
        fig : plt.Figure
            The new Figure that includes `ax`. If no new Figure created, return None.
        ax : plt.Axes
            The Axes on which the image is plotted. 
        """
        if extent is not None:
            if not isinstance(extent, list) or len(extent) != 4:
                raise TypeError('`extent` must be list of floats of length 4.')
            plot_catalogs = False
        else: extent = [0, self.rgb.shape[0], 0, self.rgb.shape[1]]

        ### Get catalogs to plot
        plot_dict = self.get_plot_catalogs(plot_catalogs)
        
        ### Plot
        fig = None
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=self.wcs[extent[0]:extent[1], extent[2]:extent[3]])
        im = ax.imshow(self.rgb[extent[0]:extent[1], extent[2]:extent[3]], origin='lower')
        for key, cat in plot_dict.items():
            coords = cat.coords
            pixel_coords = self.wcs.world_to_pixel(coords)
            ax.scatter(pixel_coords[0], pixel_coords[1], s=1, c=self.z_colors[key], label=key)
        if len(plot_dict) != 0:
            ax.legend()
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('DEC (deg)')
        return fig, ax
    
    def plot_z_hist(self, plot_catalogs=True, bins=None, ax=None):
        """
        Plot redshift distributions of catalogs in `z_catalogs` as histogram.

        Parameters
        ------------
        plot_catalogs : list(str) or bool
            List of catalog keys in `z_catalogs` specified for plotting. 
            If True, return `z_catalogs`. If False, return empty dict.
        bins : int or list or None
            Same `bins` parameter as in `plt.hist`. If None, use default bins for redshifts.
        ax : plt.Axes
            The Axes to plot image on. If None, create new figure and axes and returns both.
        
        Returns
        -------
        fig : plt.Figure
            The new Figure that includes `ax`. If no new Figure created, return None.
        ax : plt.Axes
            The Axes on which the histogram is plotted. 
        """
        ### Get catalogs to plot
        plot_dict = self.get_plot_catalogs(plot_catalogs)
        if len(plot_dict) == 0:
            logging.warning('No catalogs to plot distribution. Returning None for `fig` and `ax`.')
            return None, None

        ### Set up same bins for all hists
        if bins is None:
            bins = np.arange(0, 7, 0.1)

        ### Plot
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        for key, cat in plot_dict.items():
            zs = cat.zs
            ax.hist(zs, color=self.z_colors[key], label=key, alpha=0.5, bins=bins)
        ax.legend()
        ax.set_xlabel('Redshift z')
        ax.set_ylabel('Count')
        return fig, ax

    def add_inset(self, coords, sizes):
        """
        Add inset views of square subregions centered at given coordinates to `inset_coords`.
        If coords is outside image, ignore coords.

        Parameters
        ----------
        coords : astropy.SkyCoord or ndarray
            SkyCoord coordinates or array of shape (2,N) of x and y pixel values associated with 
            centers of square subregions. 
        sizes : array-like
            Pixel sizes of square insets for each coord in `coords`. 
        
        Returns
        -------
        result_coords : astropy.SkyCoord
            SkyCoord of added insets
        result_pixels : list
            List of 2 arrays for x and y pixel coords of added insets.
        """
        ### Convert coords to SkyCoord
        if not isinstance(coords, SkyCoord):
            try: 
                coords = self.wcs.pixel_to_world(coords[0], coords[1])
            except: raise TypeError('`coords` must be a SkyCoord or array with shape (2,N) of pixel coords.')
        try:
            if coords.size != len(sizes):
                raise ValueError('`sizes` must have same length as `coords`.')
        except: raise TypeError('`sizes` must be a list of same length as each array in `coords`')

        contained = coords.contained_by(self.wcs)
        coords = coords[contained]
        pixels = coords.to_pixel(self.wcs)
        pixels = np.array(pixels)
        self.inset_pixels = np.append(self.inset_pixels, pixels, axis=1)
        self.inset_coords = self.wcs.pixel_to_world(self.inset_pixels[0], self.inset_pixels[1])

        ### Convert sizes to units
        self.inset_sizes = np.append(self.inset_sizes, sizes[contained])
        return coords, pixels
    
    def plot_multipanel(self, plot_catalogs=True, bins=None, figsize=(20,10)):
        """
        Plot multi-panel configuration, including HUDF image and overlaid redshift objects, redshift distribution,
        and inset views of subregions.

        Parameters
        ----------
        plot_catalogs : list(str) or bool
            List of catalog keys in `z_catalogs` specified for plotting. 
            If True, return `z_catalogs`. If False, return empty dict.
        bins : int or list or None
            Same `bins` parameter as in `plt.hist`. If None, use default bins for redshifts.
        figsize : tuple(float, float)
            Same `figsize` parameter as in `plt.figure` to specify figure size.
        """
        ### Get catalogs to plot
        plot_dict = self.get_plot_catalogs(plot_catalogs)
        n = len(self.inset_sizes) # Number of insets
        n = ceil(np.sqrt(n))
        
        ### Plot 
        fig = plt.figure(figsize=figsize)       
        # Adjust gridspec size based on number of insets
        if n == 0:
            m = n
        elif n < 3:
            n = 3
            m = 2*n
        else: m = 2*n
        gs = fig.add_gridspec(n, m, hspace=0.3, wspace=0.3)
        # HUDF image on left half of figure
        imax = fig.add_subplot(gs[1:n,0:n], projection=self.wcs)
        _, imax = self.plot(plot_catalogs=plot_catalogs, ax=imax)
        imax.set_xlim(0, self.rgb.shape[0])
        # Redshift distribution above HUDF image on left side
        histax = None
        if len(plot_dict) != 0:
            histax = fig.add_subplot(gs[0,0:n])
            _, histax = self.plot_z_hist(plot_catalogs=plot_catalogs, ax=histax)

        # Plot insets
        insetax = np.zeros((n,n), dtype=object)
        for i, (x, y, size) in enumerate(zip(self.inset_pixels[0], self.inset_pixels[1], self.inset_sizes)):
            letter = chr(i+97)
            if i > 25:
                letter = i-25
            # Plot square inset on main image
            print(f'{x},{y};{size}')
            box_pixels = [x-size/2, y-size/2]
            square = Rectangle(box_pixels, size, size, fill=False, color='r')
            imax.add_patch(square)
            imax.text(box_pixels[0]+0.05*size, box_pixels[1]+0.7*size, f'({letter})', 
                      color='r')

            # Plot inset view on side
            extent = [int(box_pixels[0]), int(box_pixels[0]+size), int(box_pixels[1]), int(box_pixels[1]+size)]
            plot_ind = np.unravel_index(i, (n,n))
            ax = fig.add_subplot(gs[plot_ind[0],plot_ind[1]+n], 
                                 projection=self.wcs[extent[0]:extent[1],extent[2]:extent[3]])
            _, ax = self.plot(extent=extent, plot_catalogs=False, ax=ax)
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.tick_params(axis='y', left=False, labelleft=False)
            ax.text(0.05, 0.95, f'({letter})', ha='left', va='top',
                    color='r', bbox=dict(facecolor='white', alpha=0.8), transform=ax.transAxes)
            
            insetax[plot_ind] = ax
        
        return fig, imax, histax, insetax



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

if __name__=='__main__':
    print('Testing')