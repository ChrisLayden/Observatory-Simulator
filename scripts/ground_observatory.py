import os
import numpy as np
import pysynphot as S
from observatory import Observatory, Sensor, Telescope
import psfs

# Create a class GroundObservatory that inherits from Observatory
# Altitude is in m

class GroundObservatory(Observatory):
    def __init__(self, sensor, telescope, filter_bandpass=S.UniformTransmission(1.0),
                 exposure_time=1., num_exposures=1, seeing=1.0,
                 limiting_s_n=5., altitude=0, airmass=1.0):
        telescope.psf_type = 'gaussian'
        telescope.fwhm = seeing
        super().__init__(sensor=sensor, telescope=telescope,
                         filter_bandpass=filter_bandpass,
                         exposure_time=exposure_time,
                         num_exposures=num_exposures,
                         limiting_s_n=limiting_s_n)
        self.altitude = altitude
        self.airmass = airmass
        self.scint_noise = self.get_scint_noise()

    def get_scint_noise(self):
        diam_factor = (self.telescope.diam ** - (2/3))
        exp_time_factor = (2 * self.exposure_time) ** (-1/2)
        airmass_factor = self.airmass ** (3/2)
        altitude_factor = np.exp(-self.altitude / 8000)
        return 0.09 * diam_factor * exp_time_factor * airmass_factor * altitude_factor
    
    def observe(self, spectrum, pos=np.array([0, 0]), img_size=11, resolution=11, num_aper_frames=1):
        '''Determine the signal and noise for observation of a point source.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the intensity grid.
        pos: array-like (default [0, 0])
            The centroid position of the source on the subarray, in
            microns, with [0,0] representing the center of the
            central pixel.
        img_size: int (default 11)
            The extent of the subarray, in pixels. Should be odd, so that
            [0,0] indeed represents the center of the central pixel.
            If it isn't odd, we'll just add one.
        resolution: int (default 11)
            The number of sub-pixels per pixel. Should be odd, so that
            [0,0] represents the center of the central subpixel.
            If it isn't odd, we'll just add one.
        num_aper_frames: int (default 1)
            The number of frames to be stacked before calculating the
            optimal aperture. The default means we just find the optimal
            aperture for a single exposure.

        Returns
        -------
        results_dict: dict
            A dictionary containing the signal and noise values for the
            observation. The keys are 'signal', 'tot_noise', 'jitter_noise',
            'dark_noise', 'bkg_noise', 'read_noise', 'shot_noise', 'n_aper',
            and 'snr'.
        '''
        # For a realistic image with all noise sources, use the get_images method.
        # Here we just summarize signal and noise characteristics.
        # First find the optimal aperture. Make the subarray larger if necessary.
        aper_found = False
        while not aper_found:
            signal_grid_fine = self.signal_grid_fine(spectrum, pos, img_size, resolution)
            intrapix_grid = self.get_intrapix_grid(img_size, resolution, self.sensor.intrapix_sigma)
            signal_grid_fine *= intrapix_grid
            signal_grid = signal_grid_fine.reshape((img_size, resolution, img_size, resolution)).sum(axis=(1, 3))
            stack_image = signal_grid * num_aper_frames
            stack_pix_noise = self.single_pix_noise() * np.sqrt(num_aper_frames)
            # Relative scintillation noise over a stack decreases, because the exposure time
            # is effectively longer.
            stack_scint_noise = self.scint_noise / np.sqrt(num_aper_frames)
            optimal_aper = psfs.get_optimal_aperture(stack_image, stack_pix_noise, scint_noise=stack_scint_noise)
            aper_pads = psfs.get_aper_padding(optimal_aper)
            if min(aper_pads) > 0:
                aper_found = True
            else:
                img_size += 5
        signal = np.sum(signal_grid * optimal_aper) * self.num_exposures
        n_aper = np.sum(optimal_aper)
        shot_noise = np.sqrt(signal)
        dark_noise = np.sqrt(n_aper * self.num_exposures *
                             self.sensor.dark_current * self.exposure_time)
        bkg_noise = np.sqrt(n_aper * self.num_exposures * self.bkg_per_pix())
        read_noise = np.sqrt(n_aper * self.num_exposures * self.sensor.read_noise ** 2)
        scint_noise = signal * self.scint_noise / np.sqrt(self.num_exposures)
        tot_noise = np.sqrt(shot_noise ** 2 + dark_noise ** 2 + scint_noise ** 2 +
                            bkg_noise ** 2 + read_noise ** 2)
        results_dict = {'signal': signal, 'tot_noise': tot_noise,
                        'dark_noise': dark_noise, 'bkg_noise': bkg_noise,
                        'read_noise': read_noise, 'shot_noise': shot_noise,
                        'scint_noise': scint_noise, 'img_size': img_size,
                        'n_aper': int(n_aper), 'snr': signal / tot_noise}
        return results_dict
    

if __name__ == '__main__':
    data_folder = os.path.dirname(__file__) + '/../data/'
    sensor_bandpass = S.FileBandpass(data_folder + 'imx455.fits')
    imx455 = Sensor(pix_size=2.74, read_noise=1, dark_current=0.005,
                    full_well=51000, qe=sensor_bandpass,
                    intrapix_sigma=6)
    magellan_telescope = Telescope(diam=650, f_num=2, psf_type='gaussian', spot_size=20)
    magellan = GroundObservatory(sensor=imx455, telescope=magellan_telescope, altitude=2, airmass=1, exposure_time=0.1)
    my_spectrum = S.FlatSpectrum(20, fluxunits='abmag')
    print(magellan.observe(my_spectrum))