import numpy as np
import sys
import os
import subprocess
import astropy.io.fits as pyfits

sdss_path = 'https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/'
boss_path = 'https://data.sdss3.org/sas/dr12/boss/spectro/redux/v5_7_0/spectra/'

from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS
import astropy.units as u
import requests

class_names = ['star','galaxy', 'QSO', 'AGN'] 
## Queries für star, galaxy, quasar und AGN

query1 = "SELECT top 1000 plate, mjd, min(fiberid) as fiberid, class FROM SpecObj WHERE class = 'star' GROUP BY plate, mjd, class ORDER BY plate, mjd, class"
query2 = "SELECT top 1000 plate, mjd, min(fiberid) as fiberid, class FROM SpecObj WHERE class = 'galaxy' AND subClass != 'AGN' GROUP BY plate, mjd, class ORDER BY plate, mjd, class"
query3 = "SELECT top 1000 plate, mjd, min(fiberid) as fiberid, class FROM SpecObj WHERE class = 'QSO' AND subClass != 'AGN' GROUP BY plate, mjd, class ORDER BY plate, mjd, class"
query4 = "SELECT top 1000 plate, mjd, min(fiberid) as fiberid, class FROM SpecObj WHERE subClass = 'AGN' GROUP BY plate, mjd, class ORDER BY plate, mjd, class"

queries = [query1, query2, query3, query4]

for i in range(4):

    sdss = SDSS.query_sql(queries[i])

    speclist  = open('speclist.txt', 'w')

    for plate, mjd, fiberid in zip(sdss['plate'],sdss['mjd'],sdss['fiberid']):

        speclist.write("%04d/spec-%04d-%d-%04d.fits \n" %(plate, plate, mjd, fiberid))

    speclist.close()

    with open('speclist.txt', 'r') as f:

        names = f.readlines()

    for item in names:

        name = item[:-2]
        url = sdss_path + name
        r = requests.get(url)

        target_file = 'F:\data\spectral_fits\\' + class_names[i] + '\\' + name[5:]

        with open(target_file,'wb') as f:

            f.write(r.content)