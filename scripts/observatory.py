'''Classes and functions for synthetic photometry and noise characterization.

Classes
-------
Sensor
    Class specifying a photon-counting sensor.
Telescope
    Class specifying a telescope.
Observatory
    Class specifying a complete observatory.
'''

import os
import warnings
import psfs
import numpy as np
import pysynphot as S
from sky_background import bkg_spectrum_space
from jitter_tools import shift_values


class Sensor(object):
    '''Class specifying a photon-counting sensor.

    Attributes
    ----------
    pix_size: float
        Width of sensor pixels (assumed square), in um
    read_noise: float
        Read noise per pixel, in e-/pix
    dark_current: float
        Dark current at the sensor operating temperature,
        in e-/pix/s
    qe: pysynphot.bandpass object
        The sensor quantum efficiency as a function of wavelength
    full_well: int
        The full well (in e-) of each sensor pixel.
    '''

    def __init__(self, pix_size, read_noise, dark_current,
                 qe=S.UniformTransmission(1), full_well=100000,
                 intrapix_sigma=np.inf):
        '''Initialize a Sensor object.

        Parameters
        ----------
        pix_size: float
            Width of sensor pixels (assumed square), in um
        read_noise: float
            Read noise per pixel, in e-/pix
        dark_current: float
            Dark current at -25 degC, in e-/pix/s
        qe: pysynphot.bandpass object
            The sensor quantum efficiency as a function of wavelength
        full_well: int
            The full well (in e-) of each sensor pixel.
        intrapix_sigma: float (default np.inf)
            The standard deviation of the quantum efficiency across
            each individual pixel, in um, modeling this intrapixel
            response as a Gaussian. If not specified, the intrapixel
            response is assumed to be flat (so intrapix_sigma is infinite).
        '''

        self.pix_size = pix_size
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.full_well = full_well
        self.qe = qe
        self.intrapix_sigma = intrapix_sigma

    @property
    def qe(self):
        '''The quantum efficiency of the sensor.'''
        return self._qe

    @qe.setter
    def qe(self, value):
        if isinstance(value, float) or isinstance(value, int):
            if value < 0:
                raise ValueError('The quantum efficiency must be positive.')
            self._qe = S.UniformTransmission(value)
        elif not isinstance(value, S.spectrum.SpectralElement):
            raise ValueError('Quantum efficiency must be constant or pysynphot bandpass object.')
        else:
            self._qe = value


class Telescope(object):
    '''Class specifying a telescope.

    Attributes
    ----------
    diam: float
        Diameter of the primary aperture, in cm
    f_num: float
        Ratio of the focal length to diam
    bandpass: pysynphot.bandpass object
        The telescope bandpass as a function of wavelength,
        accounting for throughput and any geometric blocking
        factor
    focal_length: float
        The focal length of the telescope, in cm
    plate_scale: float
        The focal plate scale, in um/arcsec
    '''
    def __init__(self, diam, f_num, psf_type='airy', fwhm=None,
                 bandpass=S.UniformTransmission(1.0)):
        '''Initializing a telescope object.

        Parameters
        ----------
        diam: float
            Diameter of the primary aperture, in cm
        f_num: float
            Ratio of the focal length to diam
        psf_type: string
            The name of the PSF to use. Options are 'airy' and 'gaussian'.
        fwhm: float
            The FWHM of the psf, in arcsec. Only used if psf_type is 'gaussian'.
        bandpass: pysynphot.bandpass object
            The telescope bandpass as a function of wavelength,
            accounting for throughput and any geometric blocking
            factor

        '''

        self.diam = diam
        self.f_num = f_num
        self.bandpass = bandpass
        self.focal_length = self.diam * self.f_num
        self.psf_type = psf_type
        self.fwhm = fwhm
        if self.psf_type == 'gaussian' and self.fwhm is None:
            raise ValueError('Gaussian PSF requires a spot size.')
        self.plate_scale = 206265 / (self.focal_length * 10**4)

    @property
    def bandpass(self):
        '''The bandpass of the telescope.'''
        return self._bandpass

    @bandpass.setter
    def bandpass(self, value):
        if isinstance(value, float) or isinstance(value, int):
            if value < 0:
                raise ValueError('The quantum efficiency must be positive.')
            self._bandpass = S.UniformTransmission(value)
        elif not isinstance(value, S.spectrum.SpectralElement):
            raise ValueError('The bandpass must be a constant or a pysynphot bandpass object.')
        else:
            self._bandpass = value


class Observatory(object):
    '''Class specifying a complete observatory.'''

    def __init__(
            self, sensor, telescope, filter_bandpass=S.UniformTransmission(1.0),
            exposure_time=1., num_exposures=1, eclip_lat=90,
            limiting_s_n=5.):

        '''Initialize Observatory class attributes.

        Parameters
        ----------
        sensor: Sensor object
            The photon-counting sensor used for the observations.
        telescope: Telescope object
            The telescope used for the observations.
        filter_bandpass: pysynphot.bandpass object
            The filter bandpass as a function of wavelength.
        exposure_time: float
            The duration of each exposure, in seconds.
        num_exposures: int
            The number of exposures to stack into one image.
        eclip_lat: float
            The ecliptic latitude of the target, in degrees.
        limiting_s_n: float
            The signal-to-noise ratio constituting a detection.
        '''

        self.sensor = sensor
        self.telescope = telescope
        self.filter_bandpass = filter_bandpass
        self.bandpass = (filter_bandpass * self.telescope.bandpass *
                         self.sensor.qe)
        # Avoid error when all throughputs are flat
        all_flat_q = (np.all((self.filter_bandpass.wave is None)) and
                      np.all((self.sensor.qe.wave is None)) and
                      np.all(self.telescope.bandpass.wave is None))
        if all_flat_q:
            warnings.warn('Infinite bandpass. Manually setting wavelength limits (50-2600 nm).')
            wavelengths = S.FlatSpectrum(1, fluxunits='flam').wave
            array_bp = S.ArrayBandpass(wavelengths, np.ones(len(wavelengths)))
            self.bandpass = self.bandpass * array_bp
        self.eff_area = self.bandpass * S.UniformTransmission(np.pi * self.telescope.diam ** 2 / 4)
        self.lambda_pivot = self.bandpass.pivot()
        self.exposure_time = exposure_time
        self.num_exposures = num_exposures
        self.eclip_lat = eclip_lat
        self.limiting_s_n = limiting_s_n
        plate_scale = 206265 / (self.telescope.focal_length * 10**4)
        self.pix_scale = plate_scale * self.sensor.pix_size

    def binset(self, spectrum):
        '''Narrowest binset from telescope, sensor, filter, and spectrum.'''
        binset_list = [self.filter_bandpass.wave, self.sensor.qe.wave,
                       self.telescope.bandpass.wave, spectrum.wave]
        binset_list = [x for x in binset_list if x is not None]
        range_list = [np.ptp(x) for x in binset_list]
        obs_binset = binset_list[np.argmin(range_list)]
        return obs_binset

    def tot_signal(self, spectrum):
        '''The total number of electrons generated in one exposure.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the signal.
        '''
        S.setref(area=np.pi * self.telescope.diam ** 2 / 4)
        obs = S.Observation(spectrum, self.bandpass, binset=self.bandpass.wave,
                            force='extrap')
        raw_rate = obs.countrate()
        signal = raw_rate * self.exposure_time
        return signal

    def bkg_per_pix(self):
        '''The background noise per pixel, in e-/pix.'''
        bkg_wave, bkg_ilam = bkg_spectrum_space(self.eclip_lat)
        bkg_flam = bkg_ilam * self.pix_scale ** 2
        bkg_sp = S.ArraySpectrum(bkg_wave, bkg_flam, fluxunits='flam')
        bkg_signal = self.tot_signal(bkg_sp)
        return bkg_signal

    def eff_area_pivot(self):
        '''The effective photometric area of the observatory at the pivot wavelength, in cm^2.'''
        tele_area = np.pi * self.telescope.diam ** 2 / 4
        pivot_throughput = np.interp(self.lambda_pivot,
                                     self.bandpass.wave,
                                     self.bandpass.throughput)
        eff_area = tele_area * pivot_throughput
        return eff_area

    def psf_fwhm_um(self):
        '''The full width at half maximum of the PSF, in microns.'''
        diff_lim_fwhm = 1.025 * self.lambda_pivot * self.telescope.f_num / 10 ** 4
        if self.telescope.psf_type == 'airy':
            fwhm = diff_lim_fwhm
        elif self.telescope.psf_type == 'gaussian':
            # Need to go from cm to um
            fwhm = self.telescope.fwhm / 206265 * self.telescope.focal_length * 10 ** 4
        return fwhm

    def central_pix_frac(self):
        '''The fraction of the total signal in the central pixel.'''
        if self.telescope.psf_type == 'gaussian':
            psf_sigma = self.psf_fwhm_um() / 2.355
            half_width = self.sensor.pix_size / 2
            pix_frac = psfs.gaussian_ensq_energy(half_width, psf_sigma,
                                                 psf_sigma)
        elif self.telescope.psf_type == 'airy':
            half_width = (np.pi * self.sensor.pix_size /
                          (2 * self.telescope.f_num *
                           self.lambda_pivot * 10 ** -4))
            pix_frac = psfs.airy_ensq_energy(half_width)
        return pix_frac

    def single_pix_signal(self, spectrum):
        '''The signal within the central pixel of an image.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the signal.
        '''
        pix_frac = self.central_pix_frac()
        signal = self.tot_signal(spectrum) * pix_frac
        return signal

    def single_pix_noise(self):
        '''The noise from the background and sensor in one exposure, in e-/pix.'''
        # Noise from sensor dark current
        dark_current_noise = np.sqrt(self.sensor.dark_current *
                                     self.exposure_time)
        # Shot noise from the background
        bkg_noise = np.sqrt(self.bkg_per_pix())

        # Add noise in quadrature
        noise = np.sqrt(dark_current_noise ** 2 + bkg_noise ** 2 +
                        self.sensor.read_noise ** 2
                        )
        return noise

    def single_pix_snr(self, spectrum):
        '''The SNR for a given source with a single-pixel aperature.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the SNR.
        '''

        signal = self.single_pix_signal(spectrum)
        noise = np.sqrt(signal + self.single_pix_noise() ** 2)
        exposure_snr = signal / noise
        stack_snr = exposure_snr * np.sqrt(self.num_exposures)
        return stack_snr

    def limiting_mag(self):
        '''The limiting AB magnitude for the observatory.'''
        # We use an aperture of just 1 pixel, as this is the optimal
        # aperture for very dark objects, especially for an undersampled
        # system.
        mag_10_spectrum = S.FlatSpectrum(10, fluxunits='abmag')
        mag_10_spectrum.convert('fnu')
        mag_10_signal = self.single_pix_signal(mag_10_spectrum)
        pix_noise = self.single_pix_noise()

        def s_n_diff_mag(mag):
            '''The difference between the S/N at mag and the limiting S/N.'''
            signal = mag_10_signal * 10 ** ((10 - mag) / 2.5)
            noise = np.sqrt(signal + pix_noise ** 2)
            snr = signal / noise * np.sqrt(self.num_exposures)
            return snr - self.limiting_s_n

        # Newton-Raphson method for root-finding
        mag_tol, s_n_tol = 0.01, 0.01
        i = 1
        mag = 15
        mag_deriv_step = 0.01
        eps_mag = 1
        eps_s_n = s_n_diff_mag(mag)
        while abs(eps_s_n) > s_n_tol:
            if abs(eps_mag) < mag_tol:
                raise RuntimeError('No convergence to within 0.01 mag.')
            elif i > 20:
                raise RuntimeError('No convergence after 20 iterations.')
            eps_s_n_prime = ((s_n_diff_mag(mag + mag_deriv_step) - eps_s_n) /
                             mag_deriv_step)
            eps_mag = eps_s_n / eps_s_n_prime
            mag -= eps_mag
            eps_s_n = s_n_diff_mag(mag)
            i += 1
        return mag

    def saturating_mag(self):
        '''The saturating AB magnitude for the observatory.'''
        # We consider just the central pixel, which will saturate first.
        mag_10_spectrum = S.FlatSpectrum(10, fluxunits='abmag')
        mag_10_spectrum.convert('fnu')
        mag_10_signal = self.single_pix_signal(mag_10_spectrum)
        bkg_signal = self.bkg_per_pix()
        dark_noise = self.sensor.dark_current * self.exposure_time
        if (bkg_signal + dark_noise) >= self.sensor.full_well:
            raise ValueError('Noise itself saturates detector')

        def saturation_diff(mag):
            '''Difference between the pixel signal and full well capacity.'''
            signal = mag_10_signal * 10 ** ((10 - mag) / 2.5)
            return signal + bkg_signal + dark_noise - self.sensor.full_well

        # Newton-Raphson method for root-finding
        mag_tol, sig_tol = 0.01, 10
        i = 1
        mag = 10
        mag_deriv_step = 0.01
        eps_mag = 1
        eps_sig = saturation_diff(mag)
        while abs(eps_sig) > sig_tol:
            if abs(eps_mag) < mag_tol:
                raise RuntimeError('No convergence to within 0.01 mag.')
            elif i > 100:
                raise RuntimeError('No convergence after 100 iterations.')
            eps_sig_prime = ((saturation_diff(mag + mag_deriv_step) - eps_sig)
                             / mag_deriv_step)
            eps_mag = eps_sig / eps_sig_prime
            mag -= eps_mag
            eps_sig = saturation_diff(mag)
            i += 1
        return mag

    def signal_grid_fine(self, spectrum, pos=np.array([0, 0]),
                         img_size=11, resolution=11):
        '''The average signal (in electrons) produced across a pixel subarray.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the intensity grid.
        pos: array-like (default np.array([0, 0]))
            The centroid position of the source on the subarray, in
            microns, with [0,0] representing the center of the
            central pixel.
        img_size: int (default 11)
            The extent of the subarray, in pixels.
        resolution: int (default 11)
            The number of sub-pixels per pixel.
        '''
        tot_signal = self.tot_signal(spectrum)
        if self.telescope.psf_type == 'gaussian':
            psf_sigma = self.psf_fwhm_um() / 2.355
            cov_mat = [[psf_sigma ** 2, 0], [0, psf_sigma ** 2]]
            psf_grid = psfs.gaussian_psf(img_size, resolution,
                                         self.sensor.pix_size, pos, cov_mat)
        elif self.telescope.psf_type == 'airy':
            psf_grid = psfs.airy_disk(img_size, resolution,
                                      self.sensor.pix_size, pos,
                                      self.telescope.f_num,
                                      self.lambda_pivot)
        intensity_grid = psf_grid * tot_signal
        return intensity_grid

    def get_intrapix_grid(self, img_size=11, resolution=11, sigma=np.inf):
        '''Get the intrapixel response grid for the sensor.

        Parameters
        ----------
        img_size: int (default 11)
            The extent of the subarray, in pixels.
        resolution: int (default 11)
            The number of sub-pixels per pixel.
        sigma: float (default np.inf)
            The standard deviation of the Gaussian intrapixel response, in um.

        Returns
        -------
        intrapix_grid: array-like
            An array containing the intrapixel response across the subarray.
        '''
        if sigma == np.inf:
            intrapix_grid = np.ones((img_size * resolution, img_size * resolution))
        else:
            intrapix_single = psfs.gaussian_psf(1, resolution, self.sensor.pix_size,
                                                np.array([0, 0]),
                                                np.array([[sigma ** 2, 0],[0, sigma ** 2]]))
            intrapix_single /= np.mean(intrapix_single)
            intrapix_grid = np.tile(intrapix_single, (img_size, img_size))
        return intrapix_grid

    def get_relative_signal_grid(self, intrapix_sigma, pos=np.array([0, 0]),
                                 img_size=11, resolution=21):
        '''Get relative signal with PSF centered at each subpixel.'''
        # Looking at this grid can show you whether subpixel
        # variations in sensitivity are important.
        spec = S.FlatSpectrum(15, fluxunits='abmag')
        spec.convert('fnu')
        intrapix_grid = self.get_intrapix_grid(img_size, resolution, intrapix_sigma)
        initial_grid = self.signal_grid_fine(spec, pos, img_size, resolution)
        relative_signal_grid = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                del_x = i - resolution // 2
                del_y = j - resolution  // 2
                shifted_grid = shift_values(initial_grid, del_x, del_y) * intrapix_grid
                relative_signal_grid[i, j] = np.sum(shifted_grid)
        relative_signal_grid /= np.max(relative_signal_grid)
        return relative_signal_grid

    def get_opt_aper(self, spectrum, pos=np.array([0, 0]), img_size=11,
                     resolution=11):
        '''Find the optimal aperture for a given point source.
        
        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum of the point source to observe.
        pos: array-like (default [0, 0])
            The centroid position of the source on the central pixel,
            in microns, with [0,0] representing the center of the
            central pixel.
        img_size: int (default 11)
            The extent of the subarray, in pixels.
        resolution: int (default 11)
            The number of sub-pixels per pixel.
        '''

        aper_found = False
        while not aper_found:
            if img_size >= 50:
                raise ValueError('Subgrid is too large (>50x50 pixels).')
            initial_grid = self.signal_grid_fine(spectrum, pos, img_size, resolution)
            frame = initial_grid.reshape((img_size, resolution, img_size,
                                          resolution)).sum(axis=(1, 3))
            optimal_aper = psfs.get_optimal_aperture(frame, self.single_pix_noise())
            # Get the number of non-aperture pixels around the aperture to check
            # that the full optimal aperture is found.
            aper_pads = psfs.get_aper_padding(optimal_aper)
            if np.min(aper_pads) == 0:
                img_size += 2
            else:
                aper_found = True
        return optimal_aper

    def observe(self, spectrum, pos=np.array([0, 0]), img_size=11, resolution=11):
        '''Determine the signal and noise for observation of a point source.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the intensity grid.
        pos: array-like (default [0, 0])
            The centroid position of the source on the subarray, in
            microns, with [0,0] representing the center of the
            central pixel.
        num_frames : int (default 100)
            The number of frames to simulate.
        img_size: int (default 11)
            The extent of the subarray, in pixels. Should be odd, so that
            [0,0] indeed represents the center of the central pixel.
            If it isn't odd, we'll just add one.
        resolution: int (default 11)
            The number of sub-pixels per pixel. Should be odd, so that
            [0,0] represents the center of the central subpixel.
            If it isn't odd, we'll just add one.

        Returns
        -------
        results_dict: dict
            A dictionary containing the signal, total noise,
            dark noise, background noise, read noise, shot noise, number
            of aperture pixels, and signal-to-noise ratio.
        '''
        if img_size % 2 == 0:
            img_size += 1
        if resolution % 2 == 0:
            resolution += 1
        opt_aper = self.get_opt_aper(spectrum, pos, img_size, resolution)
        img_size = opt_aper.shape[0]
        n_aper = np.sum(opt_aper)
        initial_grid = self.signal_grid_fine(spectrum, pos, img_size, resolution)
        intrapix_sigma = self.sensor.intrapix_sigma
        intrapix_grid = self.get_intrapix_grid(img_size, resolution, intrapix_sigma)
        initial_grid *= intrapix_grid
        frame = initial_grid.reshape((img_size, resolution, img_size,
                                      resolution)).sum(axis=(1, 3))
        signal = np.sum(frame * opt_aper) * self.num_exposures
        shot_noise = np.sqrt(signal)
        dark_noise = np.sqrt(n_aper * self.num_exposures *
                             self.sensor.dark_current * self.exposure_time)
        bkg_noise = np.sqrt(n_aper * self.num_exposures * self.bkg_per_pix())
        read_noise = np.sqrt(n_aper * self.num_exposures * self.sensor.read_noise ** 2)
        tot_noise = np.sqrt(shot_noise ** 2 + dark_noise ** 2 +
                            bkg_noise ** 2 + read_noise ** 2)
        results_dict = {'signal': signal, 'tot_noise': tot_noise,
                        'dark_noise': dark_noise, 'bkg_noise': bkg_noise,
                        'read_noise': read_noise, 'shot_noise': shot_noise,
                        'n_aper': int(n_aper), 'snr': signal / tot_noise}
        return results_dict

    def get_images(self, spectrum, pos=np.array([0, 0]), num_images=1,
                   img_size=11, resolution=11, bias=100):
        '''Get realistic images of a point source.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum of the point source to observe.
        pos: array-like (default [0, 0])
            The centroid position of the source on the subarray, in
            microns, with [0,0] representing the center of the
            central pixel.
        num_images: int (default 1)
            The number of images to simulate.
        img_size: int (default 11)
            The extent of the subarray, in pixels. Should be odd, so that
            [0,0] indeed represents the center of the central pixel.
            If it isn't odd, we'll just add one.
        resolution: int (default 11)
            The number of sub-pixels per pixel. Should be odd, so that
            [0,0] represents the center of the central subpixel.
            If it isn't odd, we'll just add one.
        bias: float (default 100)
            The bias level to add to the images, in e-.

        Returns
        -------
        images: array-like
            An array containing the simulated images. These images
            are not background or bias-subtracted, nor are they
            corrected for subpixel effects.
        opt_aper: array-like
            The optimal aperture for the images.
        '''

        if img_size % 2 == 0:
            img_size += 1
        if resolution % 2 == 0:
            resolution += 1
        opt_aper = self.get_opt_aper(spectrum, pos, img_size, resolution)
        num_frames = num_images * self.num_exposures
        # Adjust img_size if it had to be increased to contain the optimal aperture
        img_size = opt_aper.shape[0]
        initial_grid = self.signal_grid_fine(spectrum, pos, img_size, resolution)
        read_signal = np.random.normal(0, self.sensor.read_noise,
                                        (img_size, img_size, num_frames))
        read_signal = np.rint(read_signal).astype(int)
        intrapix_sigma = self.sensor.intrapix_sigma
        intrapix_grid = self.get_intrapix_grid(img_size, resolution, intrapix_sigma)
        initial_grid *= intrapix_grid
        avg_frame = initial_grid.reshape((img_size, resolution, img_size,
                                          resolution)).sum(axis=(1, 3))
        images = np.zeros((num_images, img_size, img_size))
        for i in range(num_frames):
            mean_pix_bkg = (self.bkg_per_pix() + self.sensor.dark_current *
                             self.exposure_time)
            frame = np.random.poisson(avg_frame + mean_pix_bkg)
            frame = frame + read_signal[:, :, i] + bias
            images[i // self.num_exposures] += frame

        return images, opt_aper

# Some tests of Observatory behavior
if __name__ == '__main__':
    data_folder = os.path.dirname(__file__) + '/../data/'
    sensor_bandpass = S.FileBandpass(data_folder + 'imx455.fits')
    imx455 = Sensor(pix_size=2.74, read_noise=1, dark_current=0.005,
                    full_well=51000, qe=sensor_bandpass,
                    intrapix_sigma=6)

    mono_tele_v10uvs = Telescope(diam=25, f_num=1.8, psf_type='airy', bandpass=0.758)
    b_bandpass = S.ObsBandpass('johnson,b')
    r_bandpass = S.ObsBandpass('johnson,r')
    vis_bandpass = S.UniformTransmission(1.0)

    flat_spec = S.FlatSpectrum(15, fluxunits='abmag')
    flat_spec.convert('fnu')
    tess_geo_obs = Observatory(telescope=mono_tele_v10uvs, sensor=imx455,
                               filter_bandpass=r_bandpass, eclip_lat=90,
                               exposure_time=1, num_exposures=6)
    results = tess_geo_obs.observe(flat_spec, img_size=11, resolution=21)
    test_image, aper = tess_geo_obs.get_images(flat_spec, img_size=11,
                                                   resolution=21, num_images=1)
    print(results)
