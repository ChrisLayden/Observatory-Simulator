'''GUI for calculating photometry for a ground observatory'''

import os
import tkinter as tk
import pysynphot as S
import numpy as np
import matplotlib.pyplot as plt
from spectra import *
from observatory import Sensor, Telescope, Observatory
from ground_observatory import GroundObservatory
from instruments import sensor_dict_lightspeed, telescope_dict_lightspeed, filter_dict_lightspeed
from tkinter import messagebox

data_folder = os.path.dirname(__file__) + '/../data/'


class MyGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Photometry Calculations')

        PADX = 10
        PADY = 5

        # Defining sensor properties
        self.sens_header = tk.Label(self.root, text='Sensor Properties',
                                    font=['Arial', 16, 'bold'])

        self.sens_header.grid(row=0, column=0, columnspan=2,
                              padx=PADX, pady=PADY)

        self.sens_labels = []
        sens_label_names = ['Pixel Size (um)', 'Read Noise (e-/pix)',
                            'Dark Current (e-/pix/s)', 'Quantum Efficiency',
                            'Full Well Capacity']
        self.sens_boxes = []
        self.sens_vars = []
        for i, name in enumerate(sens_label_names):
            self.sens_labels.append(tk.Label(self.root,
                                             text=sens_label_names[i]))
            self.sens_labels[i].grid(row=i+2, column=0, padx=PADX, pady=PADY)
            self.sens_vars.append(tk.DoubleVar())
            self.sens_boxes.append(tk.Entry(self.root, width=10,
                                            textvariable=self.sens_vars[i]))
            self.sens_boxes[i].grid(row=i+2, column=1, padx=PADX, pady=PADY)

        self.sens_vars[0].set(5)
        self.sens_vars[1].set(3)
        self.sens_vars[2].set(0.01)
        self.sens_vars[3].set(1)
        self.sens_vars[4].set(100000)
        self.plot_qe_button = tk.Button(self.root, text='Plot QE vs. Lambda',
                                    command=self.plot_qe, fg='green')
        self.plot_qe_button.grid(row=i+3, column=0, columnspan=2, padx=PADX,
                                 pady=PADY)
        
        # If you want to select a default sensor
        self.sens_menu_header = tk.Label(self.root, text='Select Sensor',
                                         font=['Arial', 14, 'italic'])
        self.sens_menu_header.grid(row=1, column=0, columnspan=1, padx=PADX,
                                   pady=PADY)
        self.sens_options = list(sensor_dict_lightspeed.keys())
        self.sens_name = tk.StringVar()
        self.sens_menu = tk.OptionMenu(self.root, self.sens_name,
                                       *self.sens_options)
        self.sens_menu.grid(row=1, column=1, columnspan=1, padx=PADX,
                            pady=PADY)
        self.sens_name.trace_add('write', self.set_sens)
        self.sens_name.set('qCMOS')
        self.set_sens()
        for var in self.sens_vars:
            var.trace_add('write', self.clear_results)

        # Defining telescope properties
        self.tele_header = tk.Label(self.root, text='Telescope Properties',
                                    font=['Arial', 16, 'bold'])
        self.tele_header.grid(row=8, column=0, columnspan=2, padx=PADX,
                              pady=PADY)
        self.tele_labels = []
        tele_label_names = ['Diameter (cm)', 'F/number', 'Telescope Throughput', 'Altitude (m)']
        self.tele_boxes = []
        self.tele_vars = []
        for i, name in enumerate(tele_label_names):
            self.tele_labels.append(tk.Label(self.root,
                                             text=tele_label_names[i]))
            self.tele_labels[i].grid(row=i+10, column=0, padx=PADX, pady=PADY)
            self.tele_vars.append(tk.DoubleVar())
            self.tele_boxes.append(tk.Entry(self.root, width=10,
                                            textvariable=self.tele_vars[i]))
            self.tele_boxes[i].grid(row=i+10, column=1, padx=PADX, pady=PADY)

        self.tele_vars[0].set(10)
        self.tele_vars[1].set(10)
        self.tele_vars[2].set(1)
        self.tele_vars[3].set(0)

        # If you want to select a default telescope
        self.tele_menu_header = tk.Label(self.root, text='Select Telescope',
                                         font=['Arial', 14, 'italic'])
        self.tele_menu_header.grid(row=9, column=0, columnspan=1, padx=PADX,
                                   pady=PADY)
        self.tele_options = list(telescope_dict_lightspeed.keys())
        self.tele_name = tk.StringVar()
        self.tele_menu = tk.OptionMenu(self.root, self.tele_name,
                                       *self.tele_options)
        self.tele_menu.grid(row=9, column=1, columnspan=1, padx=PADX,
                            pady=PADY)
        self.tele_name.trace_add('write', self.set_tele)
        self.tele_name.trace_add('write', self.update_altitude)
        self.tele_name.set('Magellan Prototype')
        self.tele_name.trace_add('write', self.update_reimaging_throughput)
        for var in self.tele_vars:
            var.trace_add('write', self.clear_results)

        # Defining observing properties
        self.obs_header = tk.Label(self.root, text='Observing Properties',
                                   font=['Arial', 16, 'bold'])
        self.obs_header.grid(row=0, column=2, columnspan=2, padx=PADX,
                             pady=PADY)

        self.obs_labels = []
        obs_label_names = ['Exposure Time (s)', 'Exposures in Stack',
                           'Limiting SNR', 'Seeing (arcsec)',
                           'Select Filter', 'Reimaging Throughput',
                           'Object Zenith Angle (deg)', 'Lunar Phase (deg)',
                           'Object-Moon Separation (deg)']
        self.obs_boxes = []
        self.obs_vars = []
        for i, label_name in enumerate(obs_label_names):
            self.obs_labels.append(tk.Label(self.root, text=label_name))
            self.obs_labels[i].grid(row=i+1, column=2, padx=PADX, pady=PADY)
            if i == 4:
                self.obs_vars.append(tk.StringVar())
                self.obs_boxes.append(tk.OptionMenu(self.root, self.obs_vars[i],
                                         *list(filter_dict_lightspeed.keys())))
            else:
                if i == 1:
                    self.obs_vars.append(tk.IntVar())
                else:
                    self.obs_vars.append(tk.DoubleVar())
                self.obs_boxes.append(tk.Entry(self.root, width=10,
                                               textvariable=self.obs_vars[i]))
            self.obs_boxes[i].grid(row=i+1, column=3, padx=PADX, pady=PADY)

        self.obs_vars[0].set(1.0)
        self.obs_vars[1].set(1)
        self.obs_vars[2].set(5.0)
        self.obs_vars[3].set(0.5)
        self.obs_vars[4].trace_add('write', self.update_reimaging_throughput)
        self.obs_vars[4].set('Sloan g\'')
        self.obs_vars[6].set(0)
        self.obs_vars[7].set(180)
        self.obs_vars[8].set(45)
        
        for var in self.obs_vars:
            var.trace_add('write', self.clear_results)

        # Initializing labels that display results
        self.results_header = tk.Label(self.root, text='General Results',
                                       font=['Arial', 16, 'bold'])
        self.results_header.grid(row=0, column=4, columnspan=1, padx=PADX,
                                 pady=PADY)

        self.run_button_1 = tk.Button(self.root, fg='green',
                                    text='RUN',
                                    command=self.run_calcs)
        self.run_button_1.grid(row=0, column=5, columnspan=1, padx=PADX,
                             pady=PADY)

        self.results_labels = []
        results_label_names = ['Pixel Scale (arcsec/pix)',
                               'Pivot Wavelength (nm)',
                               'PSF FWHM (arcsec)',
                               'Central Pixel Ensquared Energy',
                               'A_eff at Pivot Wavelength (cm^2)', 'Limiting AB magnitude',
                               'Saturating AB magnitude', 'Airmass']
        self.results_data = []
        for i, name in enumerate(results_label_names):
            self.results_labels.append(tk.Label(self.root, text=name))
            self.results_labels[i].grid(row=i+1, column=4, padx=PADX, pady=PADY)
            self.results_data.append(tk.Label(self.root, fg='red'))
            self.results_data[i].grid(row=i+1, column=5, padx=PADX, pady=PADY)

        # Set a spectrum to observe
        self.spectrum_header = tk.Label(self.root, text='Spectrum Observation',
                                        font=['Arial', 16, 'bold'])
        self.spectrum_header.grid(row=0, column=6, columnspan=1, padx=PADX,
                                  pady=PADY)

        self.run_button_2 = tk.Button(self.root, fg='green', text='RUN',
                                    command=self.run_observation)
        self.run_button_2.grid(row=0, column=7, columnspan=1, padx=PADX,
                             pady=PADY)

        self.flat_spec_bool = tk.BooleanVar(value=True)
        self.flat_spec_check = tk.Checkbutton(self.root,
                                              text='Flat spectrum at AB mag',
                                              variable=self.flat_spec_bool)
        self.flat_spec_check.grid(row=1, column=6, padx=PADX, pady=PADY)
        self.flat_spec_mag = tk.DoubleVar(value=20.0)
        self.flat_spec_entry = tk.Entry(self.root, width=10,
                                        textvariable=self.flat_spec_mag)
        self.flat_spec_entry.grid(row=1, column=7, padx=PADX, pady=PADY)

        self.bb_spec_bool = tk.BooleanVar()
        self.bb_spec_check = tk.Checkbutton(self.root,
                                            text='Blackbody with Temp (in K)',
                                            variable=self.bb_spec_bool)
        self.bb_spec_check.grid(row=2, column=6, padx=PADX, pady=PADY)
        self.bb_temp = tk.DoubleVar()
        self.bb_spec_entry_1 = tk.Entry(self.root, width=10,
                                        textvariable=self.bb_temp)
        self.bb_spec_entry_1.grid(row=2, column=7, padx=PADX, pady=PADY)
        self.bb_dist_label = tk.Label(self.root, text='distance (in Mpc)')
        self.bb_dist_label.grid(row=3, column=6, padx=PADX, pady=PADY)
        self.bb_distance = tk.DoubleVar()
        self.bb_spec_entry_2 = tk.Entry(self.root, width=10,
                                        textvariable=self.bb_distance)
        self.bb_spec_entry_2.grid(row=3, column=7, padx=PADX, pady=PADY)
        self.bb_lbol_label = tk.Label(self.root,
                                      text='bolometric luminosity (in erg/s)')
        self.bb_lbol_label.grid(row=4, column=6, padx=PADX, pady=PADY)
        self.bb_lbol = tk.DoubleVar()
        self.bb_spec_entry_3 = tk.Entry(self.root, width=10,
                                        textvariable=self.bb_lbol)
        self.bb_spec_entry_3.grid(row=4, column=7, padx=PADX, pady=PADY)

        self.user_spec_bool = tk.BooleanVar()
        self.user_spec_check = tk.Checkbutton(self.root,
                                              text='Spectrum in spectra.py named',
                                              variable=self.user_spec_bool)
        self.user_spec_check.grid(row=5, column=6, padx=PADX, pady=PADY)
        self.user_spec_name = tk.StringVar()
        self.user_spec_entry = tk.Entry(self.root, width=20,
                                        textvariable=self.user_spec_name)
        self.user_spec_entry.grid(row=5, column=7, padx=PADX, pady=PADY)

        self.spec_results_labels = []
        spec_results_label_names = ['Signal (e-)', 'Total Noise (e-)', 'SNR',
                                    'Photometric Precision (ppm)', 'Optimal Aperture Size (pix)',
                                    'Noise Breakdown', 'Turnover Exposure Time (s)']
        self.spec_results_data = []
        for i, name in enumerate(spec_results_label_names):
            self.spec_results_labels.append(tk.Label(self.root, text=name))
            
            self.spec_results_data.append(tk.Label(self.root, fg='red'))
            # Trying to figure out spacing
            self.spec_results_labels[i].grid(row=i+6, column=6, padx=PADX, pady=PADY)
            self.spec_results_data[i].grid(row=i+6, column=7, padx=PADX, pady=PADY)

        # Make a button to plot mag vs noise
        self.plot_button = tk.Button(self.root, text='Plot Magnitude vs. Photometric Precision',
                                    command=self.plot_mag_vs_noise, fg='green')
        self.plot_button.grid(row=9, column=4, columnspan=2, padx=PADX,
                              pady=PADY)

        self.root.mainloop()

    def clear_results(self, *args):
        for label in self.results_data:
            label.config(text='')
        for label in self.spec_results_data:
            label.config(text='')

    def update_altitude(self, *args):
        # Update the altitude of the telescope based on the selected telescope
        altitude_dict = {'Magellan': 2516, 'Palomar': 1712}
        if 'Magellan' in self.tele_name.get():
            self.tele_vars[3].set(altitude_dict['Magellan'])
        # Check if name is WINTER or Hale. If either, use palomar alt
        elif 'WINTER' in self.tele_name.get() or 'Hale' in self.tele_name.get():
            self.tele_vars[3].set(altitude_dict['Palomar'])

    def update_reimaging_throughput(self, *args):
        throughput_dict_prototype = {'Sloan g\'': 0.57, 'Sloan r\'': 0.65,
                                     'Sloan i\'': 0.28, 'Sloan z\'': 0.06,
                                     'Sloan u\'': 0.05}
        throughput_dict_lightspeed = {'Sloan g\'': 0.85, 'Sloan r\'': 0.85,
                                     'Sloan i\'': 0.85, 'Sloan z\'': 0.85,
                                     'Sloan u\'': 0.85}
        if self.tele_name.get() == 'Magellan Prototype':
            throughput_dict = throughput_dict_prototype
        elif self.tele_name.get() == 'Magellan Lightspeed':
            throughput_dict = throughput_dict_lightspeed
        else:
            self.obs_vars[5].set(1.0)
            return
                                      
        if self.obs_vars[4].get() in throughput_dict.keys():
            self.obs_vars[5].set(throughput_dict[self.obs_vars[4].get()])
        else:
            self.obs_vars[5].set(1.0)

    def set_sens(self, *args):
        self.sens = sensor_dict_lightspeed[self.sens_name.get()]
        self.sens_vars[0].set(self.sens.pix_size)
        self.sens_vars[1].set(self.sens.read_noise)
        self.sens_vars[2].set(self.sens.dark_current)
        self.sens_vars[4].set(self.sens.full_well)
        if self.sens_name.get() != 'Define New Sensor':
            self.sens_vars[3] = tk.StringVar()
            self.sens_vars[3].set('ARRAY')
            self.sens_boxes[3].config(textvariable=self.sens_vars[3])
            self.sens_boxes[3].config(state='disabled')
        else:
            self.sens_vars[3] = tk.DoubleVar()
            self.sens_boxes[3].config(textvariable=self.sens_vars[3])
            self.sens_boxes[3].config(state='normal')
            self.sens_vars[3].set(np.mean(self.sens.qe.throughput))

    def set_tele(self, *args):
        self.tele = telescope_dict_lightspeed[self.tele_name.get()]
        self.tele_vars[0].set(self.tele.diam)
        self.tele_vars[1].set(self.tele.f_num)
        self.tele_vars[2].set(self.tele.bandpass)

    def set_obs(self):
        sens_vars = [i.get() for i in self.sens_vars]
        if sens_vars[3] == 'ARRAY':
            sens_vars[3] = self.sens.qe
        else:
            sens_vars[3] = S.UniformTransmission(float(sens_vars[3]))
        sens = Sensor(*sens_vars)
        tele_vars = [i.get() for i in self.tele_vars]
        tele_vars[2] = S.UniformTransmission(tele_vars[2])
        tele = Telescope(*tele_vars)
        exposure_time = self.obs_vars[0].get()
        num_exposures = int(self.obs_vars[1].get())
        limiting_snr = self.obs_vars[2].get()
        filter_bp = filter_dict_lightspeed[self.obs_vars[4].get()]
        reimaging_throughput = self.obs_vars[5].get()
        filter_bp = S.UniformTransmission(reimaging_throughput) * filter_bp
        seeing_arcsec = self.obs_vars[3].get()
        obs_zo = self.obs_vars[6].get()
        obs_altitude = self.tele_vars[3].get()
        obs_alpha = self.obs_vars[7].get()
        obs_rho = self.obs_vars[8].get()
        observatory = GroundObservatory(sens, tele, exposure_time=exposure_time,
                                        num_exposures=num_exposures,
                                        limiting_s_n=limiting_snr,
                                        filter_bandpass=filter_bp,
                                        seeing=seeing_arcsec,
                                        zo=obs_zo, rho=obs_rho,
                                        altitude=obs_altitude,
                                        alpha=obs_alpha)
        return observatory

    def set_spectrum(self):
        if self.flat_spec_bool.get():
            abmag = self.flat_spec_mag.get()
            # Convert to Jansky's; sometimes Pysynphot freaks out when
            # using AB magnitudes.
            fluxdensity_Jy = 10 ** (-0.4 * (abmag - 8.90))
            spectrum = S.FlatSpectrum(fluxdensity=fluxdensity_Jy,
                                      fluxunits='Jy')
            # spectrum.convert('fnu')
        elif self.bb_spec_bool.get():
            temp = self.bb_temp.get()
            distance = self.bb_distance.get()
            l_bol = self.bb_lbol.get()
            spectrum = blackbody_spec(temp, distance, l_bol)
        elif self.user_spec_bool.get():
            spectrum_name = self.user_spec_name.get()
            spectrum = eval(spectrum_name)
        else:
            raise ValueError('No spectrum specified')
        return spectrum

    def run_calcs(self):
        try:
            observatory = self.set_obs()
            limiting_mag = observatory.limiting_mag()
            saturating_mag = observatory.saturating_mag()

            self.results_data[0].config(text=format(observatory.pix_scale, '4.3f'))
            self.results_data[1].config(text=format(observatory.lambda_pivot / 10, '4.1f'))
            psf_fwhm_arcsec = observatory.psf_fwhm_um() * observatory.pix_scale / observatory.sensor.pix_size
            self.results_data[2].config(text=format(psf_fwhm_arcsec, '4.3f'))
            self.results_data[3].config(text=format(100 * observatory.central_pix_frac(),
                                                    '4.1f') + '%')
            self.results_data[4].config(text=format(observatory.eff_area_pivot(), '4.2f'))
            self.results_data[5].config(text=format(limiting_mag, '4.3f'))
            self.results_data[6].config(text=format(saturating_mag, '4.3f'))
            self.results_data[7].config(text=format(observatory.airmass, '4.3f'))
        except ValueError as inst:
            messagebox.showerror('Value Error', inst)

    def run_observation(self):
        try:
            spectrum = self.set_spectrum()
            observatory = self.set_obs()
            results = observatory.observe(spectrum)
            signal = int(results['signal'])
            noise = int(results['tot_noise'])
            snr = results['signal'] / results['tot_noise']
            phot_prec = 10 ** 6 / snr
            turnover_exp_time = observatory.turnover_exp_time(spectrum)
            self.spec_results_data[0].config(text=format(signal, '4d'))
            self.spec_results_data[1].config(text=format(noise, '4d'))
            self.spec_results_data[2].config(text=format(snr, '4.3f'))
            self.spec_results_data[3].config(text=format(phot_prec, '4.3f'))
            self.spec_results_data[4].config(text=format(results['n_aper'], '2d'))
            noise_str = ('Shot noise: ' + format(results['shot_noise'], '.2f') +
                         '\nDark noise: ' + format(results['dark_noise'], '.2f') +
                         '\nRead noise: ' + format(results['read_noise'], '.2f') +
                         '\nBackground noise: ' + format(results['bkg_noise'], '.2f') +
                         '\nScintillation noise: ' + format(results['scint_noise'], '.2f'))
            self.spec_results_data[5].config(text=noise_str)
            self.spec_results_data[6]. config(text=format(turnover_exp_time, '4.3f'))

        except ValueError as inst:
            messagebox.showerror('Value Error', inst)

    def plot_qe(self):
        qe = self.sens.qe
        # Check if uniform transmission
        if isinstance(qe, S.UniformTransmission):
            wave = np.linspace(200, 1000, 100)
            throughput = np.ones_like(wave) * self.sens_vars[3].get()
            plt.plot(wave, throughput * 100)
        else:
            plt.plot(qe.wave / 10, qe.throughput * 100)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Quantum Efficiency (%)')
        plt.show()

    def plot_mag_vs_noise(self):
        mag_points = np.linspace(10, 28, 15)
        ppm_points = np.zeros_like(mag_points)
        ppm_points_source = np.zeros_like(mag_points)
        ppm_points_read = np.zeros_like(mag_points)
        ppm_points_bkg = np.zeros_like(mag_points)
        ppm_points_dc = np.zeros_like(mag_points)
        ppm_points_scint = np.zeros_like(mag_points)
        observatory = self.set_obs()
        img_size = 11
        for i, mag in enumerate(mag_points):
            spectrum = S.FlatSpectrum(mag, fluxunits='abmag')
            results = observatory.observe(spectrum, img_size=img_size)
            snr = results['signal'] / results['tot_noise']
            phot_prec = 10 ** 6 / snr
            ppm_points[i] = phot_prec
            ppm_points_source[i] = 10 ** 6 * results['shot_noise'] / results['signal']
            ppm_points_read[i] = 10 ** 6 * results['read_noise'] / results['signal']
            ppm_points_bkg[i] = 10 ** 6 * results['bkg_noise'] / results['signal']
            ppm_points_dc[i] = 10 ** 6 * results['dark_noise'] / results['signal']
            ppm_points_scint[i] = 10 ** 6 * results['scint_noise'] / results['signal']
            img_size = results['img_size']
        plt.plot(mag_points, ppm_points, label='Total Noise')
        plt.plot(mag_points, ppm_points_source, label='Shot Noise')
        plt.plot(mag_points, ppm_points_read, label='Read Noise')
        plt.plot(mag_points, ppm_points_bkg, label='Background Noise')
        plt.plot(mag_points, ppm_points_dc, label='Dark Current Noise')
        plt.plot(mag_points, ppm_points_scint, label='Scintillation Noise')
        ppm_threshold = 1e6 / self.obs_vars[2].get()
        plt.fill_between(np.linspace(10, 30, 10), ppm_threshold, 2e6, color='red', alpha=0.1,
                         label='Non-detection')
        plt.xlim(10, 28)
        # Just set top upper limit
        plt.ylim(1, 2e6)
        plt.xlabel('AB Magnitude')
        plt.ylabel('Photometric Precision (ppm)')
        plt.yscale('log')
        plt.legend()
        # Make title text with telescope name, bandbpass, and exposure time
        tele_name = self.tele_name.get()
        bandpass = self.obs_vars[4].get()
        exposure_time = self.obs_vars[0].get()
        title = f'{tele_name}, {bandpass}, t_exp={exposure_time}s'
        plt.title(title)
        plt.show()


MyGUI()
