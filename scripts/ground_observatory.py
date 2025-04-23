'''Subclass of Observatory for ground-based telescopes.

Classes
-------
GroundObservatory
    A class for ground-based observatories, inheriting from Observatory.
    Key differences are the addition of scintillation noise (using airmass
    and altitude values), atmospheric seeing, and the effects of the moon
    on the sky background (ZZZ that's not yet added).
'''


import os
import numpy as np
import pysynphot as S
from observatory import Observatory, Sensor, Telescope
from sky_background import bkg_spectrum_ground
import psfs

class GroundObservatory(Observatory):
    '''A class for ground-based observatories, inheriting from Observatory.'''

    def __init__(self, sensor, telescope, filter_bandpass=S.UniformTransmission(1.0),
                 exposure_time=1., num_exposures=1, seeing=1.0,
                 limiting_s_n=5., altitude=0, alpha=180, zo=0, rho=45):
        '''Initialize the GroundObservatory class.
        
        Parameters
        ----------
        sensor: Sensor
            The sensor object to be used for observations.
        telescope: Telescope
            The telescope object to be used for observations.
        filter_bandpass: pysynphot.spectrum object
            The bandpass filter to be used for observations.
        exposure_time: float
            The exposure time for observations, in seconds.
        num_exposures: int
            The number of exposures to be taken.
        seeing: float
            The PSF FWHM, in arcseconds. Assumes a Gaussian PSF, and that
            all broadening is due to the atmosphere.
        limiting_s_n: float
            The limiting signal-to-noise ratio for observations.
        altitude: float
            The altitude of the observatory, in meters.
        alpha: float
            The lunar phase angle, in degrees. 0 is full moon, 180 is new moon.
            Should only be between 0 and 180 (assumes symmetry between
            waning/waxing).
        zo: float
            The zenith angle of the object being observed, in degrees.
        rho: float
            The angular separation between the moon and the object, in degrees.
            For simplicity, we assume the moon is lower in the sky than the
            object, with zenith angle zo - rho.
        '''

        super().__init__(sensor=sensor, telescope=telescope,
                         filter_bandpass=filter_bandpass,
                         exposure_time=exposure_time,
                         num_exposures=num_exposures,
                         limiting_s_n=limiting_s_n)

        telescope.psf_type = 'gaussian'
        telescope.fwhm = seeing
        self.alpha = alpha
        self.altitude = altitude
        self.zo = zo
        # Formula 3 in Krisciunas & Schaefer 1991 for airmass.
        self.airmass = (1 - 0.96 * np.sin(np.radians(zo)) ** 2) ** -0.5
        self.rho = rho
        # The zenith angle of the moon, in degrees.
        self.zm = zo - rho
        self.scint_noise = self.get_scint_noise()
        
    @property
    def alpha(self):
        '''The lunar phase angle during observation.'''
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value < 0 or value > 180:
            raise ValueError("alpha must be between 0 and 180 degrees.")
        else:
            self._alpha = value

    def get_scint_noise(self):
        '''Calculate the scintillation noise, per formula from Young, 1967.'''
        diam_factor = self.telescope.diam ** - (2/3)
        exp_time_factor = (2 * self.exposure_time) ** (-1/2)
        airmass_factor = self.airmass ** (3/2)
        altitude_factor = np.exp(-self.altitude / 8000)
        return 0.09 * diam_factor * exp_time_factor * airmass_factor * altitude_factor
    
    def bkg_per_pix(self):
        '''The background noise per pixel, in e-/pix.'''
        bkg_wave, bkg_ilam = bkg_spectrum_ground(alpha=self.alpha, rho=self.rho,
                                                 Zm=self.zm, Zo=self.zo,)
        bkg_flam = bkg_ilam * self.pix_scale ** 2
        bkg_sp = S.ArraySpectrum(bkg_wave, bkg_flam, fluxunits='flam')
        bkg_signal = self.tot_signal(bkg_sp)
        return bkg_signal
    
    def turnover_exp_time(self, spectrum, eps=0.02):
        '''Get the exposure time at which background noise is equal to read noise.'''
        exp_time_initial = self.exposure_time
        new_exp_time = exp_time_initial
        results_dict = self.observe(spectrum)
        read_over_bkg = results_dict['read_noise'] / results_dict['bkg_noise']
        i = 0
        while abs(read_over_bkg - 1) > eps and i < 10:
            new_exp_time = self.exposure_time * read_over_bkg
            self.exposure_time = new_exp_time
            results_dict = self.observe(spectrum)
            read_over_bkg = results_dict['read_noise'] / results_dict['bkg_noise']
            i += 1
        if i == 10:
            raise ValueError("Turnover exposure time not found within 10 iterations.")
        self.exposure_time = exp_time_initial
        return new_exp_time

    def observe(self, spectrum, pos=np.array([0, 0]), img_size=11,
                resolution=11, num_aper_frames=1):
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
            signal_grid = signal_grid_fine.reshape((img_size, resolution,
                                                    img_size, resolution)).sum(axis=(1, 3))
            stack_image = signal_grid * num_aper_frames
            stack_pix_noise = self.single_pix_noise() * np.sqrt(num_aper_frames)
            # Relative scintillation noise over a stack decreases, because the exposure time
            # is effectively longer.
            stack_scint_noise = self.scint_noise / np.sqrt(num_aper_frames)
            optimal_aper = psfs.get_optimal_aperture(stack_image, stack_pix_noise,
                                                     scint_noise=stack_scint_noise)
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
    magellan_telescope = Telescope(diam=650, f_num=2)
    magellan = GroundObservatory(sensor=imx455, telescope=magellan_telescope,
                                 altitude=2, exposure_time=0.1,
                                 seeing=0.5, alpha=180, zo=0, rho=45)
    my_spectrum = S.FlatSpectrum(20, fluxunits='abmag')
    print(magellan.observe(my_spectrum))
    print(magellan.turnover_exp_time(my_spectrum))
    
