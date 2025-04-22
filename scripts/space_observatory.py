'''Subclass of Observatory for space-based telescopes.

Classes
-------
SpaceObservatory
    A class for space-based observatories, inheriting from Observatory.
    Key difference is the addition of jitter noise, when a jitter PSD
    is defined.
'''


import os
import numpy as np
import pysynphot as S
from observatory import Observatory, Sensor, Telescope
import psfs
from jitter_tools import jittered_array, integrated_stability, get_pointings, shift_values

class SpaceObservatory(Observatory):
    '''A class for space-based observatories, inheriting from Observatory.'''

    def __init__(self, sensor, telescope, filter_bandpass=S.UniformTransmission(1.0),
                 exposure_time=1., num_exposures=1, limiting_s_n=5.0,
                 eclip_lat=90, jitter_psd=None):
        '''Initialize the SpaceObservatory class.
        
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
        limiting_s_n: float
            The limiting signal-to-noise ratio for observations.
        eclip_lat: float
            The ecliptic latitude of the object being observed, in deg.
        jitter_psd: array-like (n x 2)
            The power spectral density of the jitter. Assumes PSD is
            the same for x and y directions, and no roll jitter.
            Contains two columns: the first is the frequency, in Hz, and
            the second is the PSD, in arcseconds^2/Hz. If not specified,
            it's assumed that there is no pointing jitter.
        '''

        super().__init__(sensor=sensor, telescope=telescope,
                         filter_bandpass=filter_bandpass,
                         exposure_time=exposure_time,
                         num_exposures=num_exposures,
                         limiting_s_n=limiting_s_n,
                         eclip_lat=eclip_lat)
        self.jitter_psd = jitter_psd
        # The total power in the jitter PSD, measured as a 1-sigma jitter,
        # in arcseconds. The effective stability of images can be reduced
        # by sampling at higher frequencies.
        if self.jitter_psd is not None:
            self.stability = integrated_stability(10, jitter_psd[:, 0],
                                                  jitter_psd[:, 1])
        else:
            self.stability = 0

    def get_pointings(self, num_frames=100, resolution=11):
        '''Get time series jitter data.
        
        Parameters
        ----------
        num_frames: int (default 100)
            The number of frames for which to calculate pointings.
        resolution: int (default 11)
            The number of subpixels per pixel in the subgrid.
        
        Returns
        -------
        pointings_array: array-like
            An array containing the pointings for each frame, in arcsec.
        '''

        if self.jitter_psd is not None:
            jitter_psd_pix = self.jitter_psd.copy()
            jitter_psd_pix[:, 1] = jitter_psd_pix[:, 1] / self.pix_scale ** 2
            jitter_time = np.min([self.exposure_time / 10.0, 0.5])
            pointings_array = get_pointings(self.exposure_time, num_frames,
                                            jitter_time, resolution, jitter_psd_pix)
        else:
            jitter_time = self.exposure_time
            pointings_array = np.zeros((num_frames, 1, 2), dtype=int)
        return pointings_array

    def get_remaining_jitter(self, resolution=11):
        '''Find the power remaining in the jitter beyond the image sampling frequency.'''
        if self.jitter_psd is not None:
            jitter_time = np.min([self.exposure_time / 100.0, 0.5])
            tot_power = np.sqrt(np.trapz(self.jitter_psd[:, 1], self.jitter_psd[:, 0]))
            jitter_time = self.exposure_time
            pointings_array = get_pointings(self.exposure_time, 500,
                                            jitter_time, resolution, self.jitter_psd)
            pointings_x = pointings_array[:, :, 0].flatten() / resolution
            pointings_y = pointings_array[:, :, 1].flatten() / resolution
            removed_power = (np.std(pointings_x) + np.std(pointings_y)) / 2
            remaining_power = np.sqrt(tot_power ** 2 - removed_power ** 2)
        else:
            remaining_power = 0
        return remaining_power

    def get_opt_aper(self, spectrum, pos=np.array([0, 0]), img_size=11,
                     resolution=11, num_aper_frames=200):
        '''Find the optimal aperture for a given jittered point source.

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
        num_aper_frames: int (default 200)
            The number of frames stacked to find the optimal aperture.
            It is important to stack a large number of frames to
            fully smear out the jitter in all directions.
        '''
        pointings_array = self.get_pointings(num_aper_frames, resolution)
        max_shift = np.max(abs(pointings_array))
        img_size = np.max([img_size, int(max_shift / resolution) * 2 + 1])
        aper_found = False
        while not aper_found:
            if img_size >= 50:
                raise ValueError('Subgrid is too large (>50x50 pixels).')
            initial_grid = self.signal_grid_fine(spectrum, pos, img_size, resolution)
            stack_image = np.zeros((img_size, img_size))
            for i in range(num_aper_frames):
                pointings = pointings_array[i]
                avg_grid = jittered_array(initial_grid, pointings)
                frame = avg_grid.reshape((img_size, resolution, img_size,
                                        resolution)).sum(axis=(1, 3))
                # Find the average shift caused by jitter and shift the frame back
                # by that amount. On board, this would require a centroiding algorithm
                # or a feed-in from the star tracker.
                shift_x = np.rint(np.mean(pointings[:,0]) / resolution).astype(int)
                shift_y = np.rint(np.mean(pointings[:,1]) / resolution).astype(int)
                frame = shift_values(frame, -shift_x, -shift_y)
                stack_image += frame
            stack_pix_noise = self.single_pix_noise() * np.sqrt(num_aper_frames)
            optimal_aper = psfs.get_optimal_aperture(stack_image, stack_pix_noise)
            # Get the number of non-aperture pixels around the aperture to check
            # that no pixels in or adjacent to the aperture left the subarray during
            # jitter. If any did, make the subarray larger and try again.
            aper_pads = psfs.get_aper_padding(optimal_aper)
            max_falloff = np.ceil(max_shift / resolution - np.min(aper_pads) + 1).astype(int)
            if max_falloff > 0:
                img_size += 2 * max_falloff
            else:
                aper_found = True
        return optimal_aper

    def observe(self, spectrum, pos=np.array([0, 0]), num_frames=500,
                img_size=11, resolution=11, subpix_correct=False):
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
            The number of frames to simulate with jitter.
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
            A dictionary containing the signal, total noise, jitter noise,
            dark noise, background noise, read noise, shot noise, number
            of aperture pixels, and signal-to-noise ratio.
        '''
        if img_size % 2 == 0:
            img_size += 1
        if resolution % 2 == 0:
            resolution += 1
        opt_aper = self.get_opt_aper(spectrum, pos, img_size, resolution)
        aper_pads = psfs.get_aper_padding(opt_aper)
        pointings_array = self.get_pointings(num_frames, resolution)
        # Make subarray larger if jitter causes any aperture subpixels
        # to fall off the subarray.
        max_shift = np.max(abs(pointings_array))
        max_falloff = np.ceil(max_shift / resolution - np.min(aper_pads)).astype(int)
        if max_falloff >= 0:
            opt_aper = np.pad(opt_aper, max_falloff, 'constant')
        # Adjust img_size if it had to be increased to contain the optimal aperture
        img_size = opt_aper.shape[0]
        n_aper = np.sum(opt_aper)
        # Don't add source shot, background, dark current, or read noise, because
        # we already know their effect. Here we want to isolate jitter noise.
        # For a realistic image with all noise sources, use the get_images method.
        initial_grid = self.signal_grid_fine(spectrum, pos, img_size, resolution)
        signal_list = np.zeros(num_frames)
        intrapix_sigma = self.sensor.intrapix_sigma
        intrapix_grid = self.get_intrapix_grid(img_size, resolution, intrapix_sigma)
        rel_signal_grid = self.get_relative_signal_grid(intrapix_sigma, pos, img_size, resolution)
        shifts_x = np.zeros(num_frames)
        center_fracs = np.zeros(num_frames)
        for i in range(num_frames):
            pointings = pointings_array[i]
            avg_grid = jittered_array(initial_grid, pointings)
            avg_grid *= intrapix_grid
            frame = avg_grid.reshape((img_size, resolution, img_size,
                                      resolution)).sum(axis=(1, 3))
            # Find the average shift caused by the jitter and shift the frame
            # by the same amount. On board, this would require a centroiding algorithm
            # or a feed-in from the star tracker.
            shift_x = np.rint(np.mean(pointings[:,0]) / resolution).astype(int)
            shifts_x[i] = shift_x
            shift_y = np.rint(np.mean(pointings[:,1]) / resolution).astype(int)
            frame = shift_values(frame, -shift_x, -shift_y)
            center_fracs[i] = frame.max() / frame.sum()
            frame_signal = np.sum(frame * opt_aper)
            if subpix_correct:
                subpix_shift_x = np.rint(np.mean(pointings[:,0]) - shift_x * resolution)
                subpix_shift_y = np.rint(np.mean(pointings[:,1]) - shift_y * resolution)
                # If shift is right on a pixel edge (-resolution/2 or resolution/2),
                # set to the nearest subpixel.
                subpix_shift_x = np.clip(subpix_shift_x, -resolution // 2, resolution // 2)
                subpix_shift_y = np.clip(subpix_shift_y, -resolution // 2, resolution // 2)
                rel_signal_index_x = int(subpix_shift_x) + resolution // 2
                rel_signal_index_y = int(subpix_shift_y) + resolution // 2
                flux_loss_factor = rel_signal_grid[rel_signal_index_x, rel_signal_index_y]
                frame_signal /= flux_loss_factor
            signal_list[i] = frame_signal
        signal = np.mean(signal_list) * self.num_exposures
        jitter_noise = np.std(signal_list) * np.sqrt(self.num_exposures)
        shot_noise = np.sqrt(signal)
        dark_noise = np.sqrt(n_aper * self.num_exposures *
                             self.sensor.dark_current * self.exposure_time)
        bkg_noise = np.sqrt(n_aper * self.num_exposures * self.bkg_per_pix())
        read_noise = np.sqrt(n_aper * self.num_exposures * self.sensor.read_noise ** 2)
        tot_noise = np.sqrt(jitter_noise ** 2 + shot_noise ** 2 + dark_noise ** 2 +
                            bkg_noise ** 2 + read_noise ** 2)
        results_dict = {'signal': signal, 'tot_noise': tot_noise,
                        'jitter_noise': jitter_noise,
                        'dark_noise': dark_noise, 'bkg_noise': bkg_noise,
                        'read_noise': read_noise, 'shot_noise': shot_noise,
                        'n_aper': int(n_aper), 'snr': signal / tot_noise}
        return results_dict

    def get_images(self, spectrum, pos=np.array([0, 0]), num_images=1,
                   img_size=11, resolution=11, bias=100):
        '''Get realistic images of a point source with jitter.

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
        subpix_correct: bool (default False)
            Whether to correct for subpixel sensitivity variations.
        bias: float (default 100)
            The bias level to add to the images, in e-.

        Returns
        -------
        images: array-like
            An array containing the simulated images. These images
            are not background or bias-subtracted, nor are they
            corrected for subpixel jitter.
        opt_aper: array-like
            The optimal aperture for the images.
        '''

        if img_size % 2 == 0:
            img_size += 1
        if resolution % 2 == 0:
            resolution += 1
        opt_aper = self.get_opt_aper(spectrum, pos, img_size, resolution)
        aper_pads = psfs.get_aper_padding(opt_aper)
        num_frames = num_images * self.num_exposures
        pointings_array = self.get_pointings(num_frames, resolution)
        # Make subarray larger if jitter causes any aperture subpixels
        # to fall off the subarray.
        max_shift = np.max(abs(pointings_array))
        max_falloff = np.ceil(max_shift / resolution - np.min(aper_pads)).astype(int)
        if max_falloff >= 0:
            opt_aper = np.pad(opt_aper, max_falloff, 'constant')
        # Adjust img_size if it had to be increased to contain the optimal aperture
        img_size = opt_aper.shape[0]
        initial_grid = self.signal_grid_fine(spectrum, pos, img_size, resolution)
        read_signal = np.random.normal(0, self.sensor.read_noise,
                                        (img_size, img_size, num_frames))
        read_signal = np.rint(read_signal).astype(int)
        intrapix_sigma = self.sensor.intrapix_sigma
        intrapix_grid = self.get_intrapix_grid(img_size, resolution, intrapix_sigma)
        images = np.zeros((num_images, img_size, img_size))
        for i in range(num_frames):
            pointings = pointings_array[i]
            avg_grid = jittered_array(initial_grid, pointings)
            avg_grid *= intrapix_grid
            frame = avg_grid.reshape((img_size, resolution, img_size,
                                      resolution)).sum(axis=(1, 3))
            # Find the average shift caused by the jitter in the frame and shift the frame
            # back by that amount. On board, this would require a centroiding algorithm
            # or a feed-in from the star tracker.
            shift_x = np.rint(np.mean(pointings[:,0]) / resolution).astype(int)
            shift_y = np.rint(np.mean(pointings[:,1]) / resolution).astype(int)
            frame = shift_values(frame, -shift_x, -shift_y)
            frame = np.random.poisson(frame + self.mean_pix_bkg)
            frame = frame + read_signal[:, :, i] + bias
            images[i // self.num_exposures] += frame

        return images, opt_aper


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
    tess_psd = np.genfromtxt(data_folder + 'TESS_Jitter_PSD.csv', delimiter=',')
    tess_geo_obs = SpaceObservatory(telescope=mono_tele_v10uvs, sensor=imx455,
                               filter_bandpass=r_bandpass, eclip_lat=90,
                               exposure_time=1, num_exposures=6,
                               jitter_psd=tess_psd)
    results = tess_geo_obs.observe(flat_spec, num_frames=500, img_size=21,
                                   resolution=21, subpix_correct=True)
    print(results)
