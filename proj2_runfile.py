import numpy as np
import matplotlib.pyplot as plt
import pyvo as vo
from IPython.display import Image as ipImage, display
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm
import matplotlib as mpl
import sys
from astropy.visualization import make_lupton_rgb
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)



if __name__=='__main__':
    hudf = []
    hudf_header = []
    print('Loading Hubble UDF fits files...')
    for i, c in enumerate(['i', 'v', 'b']):
        txt = f'\r    Loading {c} image'
        sys.stdout.write(txt)
        sys.stdout.flush()
        with fits.open(f'data/h_udf_wfc_{c}_drz_img.fits') as hdul:
            hudf.append(hdul[0].data)
            hudf_header.append(hdul[0].header)
    print('')
    print('Finished Loading')



    plt.imshow(hudf[0], origin='lower', cmap=mpl.colormaps['Reds'], vmin=0.00001, vmax=.002, alpha=0.5)
    plt.imshow(hudf[1], origin='lower', cmap=mpl.colormaps['Greens'], vmin=0.00001, vmax=.002, alpha=0.5)
    plt.imshow(hudf[2], origin='lower', cmap=mpl.colormaps['Blues'], vmin=0.00001, vmax=.002, alpha=0.5)
    plt.show()