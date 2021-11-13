import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import os
import csv

########## Input ##########

fits_path = 'F:\\data\\spectral_fits\\'
data_path = 'F:\\data\\'

samples_per_class = 1000

smallest_wavelength = 4000 # in Angström
biggest_wavelength = 9000 

########## Program ##########

all_flux = []

for directory in os.listdir(fits_path):
    
    count_failed=0
    count_added=0
    
    for filename in os.listdir(fits_path + directory + '\\'):
        
        path = fits_path + '\\' + directory + '\\' + filename
        
        # fits-Dateien öffnen und wavelength + flux einlesen
        hdul = fits.open(path)
        data = hdul[1].data
        flux = data['flux']
        wavelength = 10**data['loglam']
        hdul.close()
        
        # first und last Index finden
        for i in range(len(wavelength)):
            if wavelength[i]>smallest_wavelength:
                first_index = i
                break

        for i in range(len(wavelength)):
            if wavelength[i]>biggest_wavelength:
                last_index = i
                break

        # wavenlength und flux Listen schneiden
        wavelength = wavelength[first_index:last_index]
        flux = flux[first_index:last_index]
        
        if len(wavelength) == 3522 and count_added < samples_per_class:
            all_flux.append(flux) 
            count_added += 1        

        if len(wavelength) != 3522:
            print("Länge der Liste wavelength ist: " + str(len(wavelength)))
            count_failed += 1            
            
    print(str(count_failed/1000*100) + "% waren nicht erfolgreich bei der Klasse:" + directory)

# Numpy Arrays mit Daten füllen
data = np.array(all_flux)

labels = np.zeros(shape=(4000,), dtype='int')
for i in range(4):
    for t in range(samples_per_class):
        labels[i*1000+t] = i
        
wavelengths = np.array(wavelength)

# Numpy arrays in .npy Dateien speichern
np.save(data_path + "data.npy", data)
np.save(data_path + "labels.npy", labels)
np.save(data_path + "wavelengths.npy", wavelengths)