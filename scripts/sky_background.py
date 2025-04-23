'''Functions to calculate the sky background spectrum

Functions
---------
bkg_ilam : float
    Return the specific intensity of sky background light
    at a given wavelength and ecliptic latitude.
bkg_spectrum : array-like
    Return the spectrum of light from the sky background
    at a given ecliptic latitude.
'''

import os
import numpy as np

abs_path = os.path.dirname(__file__)
# Data for background light, as observed from space via Hubble
# Log(specific intensity) of the zodiacal light at ecliptic latitude 90 deg,
eclip_ilam = np.genfromtxt(abs_path + '/../data/ZodiacalLight.csv',
                           delimiter=',')
eclip_ilam[:, 1] = 10 ** eclip_ilam[:, 1]
# The specific intensity for a V-band baseline
eclip_ilam_v = np.interp(5500, eclip_ilam[:, 0], eclip_ilam[:, 1])


def bkg_ilam(lam, eclip_angle):
    '''Return the specific intensity of sky background light.

    Parameters
    ----------
    lam : float
        The wavelength of the light, in Angstroms.
    eclip_angle : float
        The ecliptic latitude, in degrees. We assume the specific intensity
        scales with b in the same way as it does for zodiacal light in the
        V-band. This is conservative for most other bands, especially the UV,
        for which most background light comes from diffuse galactic light.
    Returns
    -------
    ilam : float
        The specific intensity of the sky background, in
        erg/s/cm^2/Ang/arcsec^2.
    '''
    # Uses a linear fit to the magnitude of zodiacal light in the V-band as a
    # function of ecliptic latitude described in Sullivan et al. 2015
    vmag_max = 23.345
    del_vmag = 1.148
    # The V-band magnitude, in mag/arcsec^2
    vmag = vmag_max - del_vmag * ((eclip_angle - 90) / 90) ** 2
    # The V-band specific intensity, in erg/s/cm^2/Hz/arcsec^2
    inu_v = 10 ** (-vmag / 2.5) * 3631 * 10 ** -23
    # Make sure to use c in Angstroms
    ilam_v = inu_v * (3 * 10 ** 18) / lam ** 2
    freq_factor = (np.interp(lam, eclip_ilam[:, 0], eclip_ilam[:, 1]) /
                   eclip_ilam_v)
    ilam = ilam_v * freq_factor
    return ilam


def bkg_spectrum_space(eclip_angle=90):
    '''Return the spectrum of light from the sky background.

    Parameters
    ----------
    eclip_angle : float
        The ecliptic latitude, in degrees.
    Returns
    -------
    spectrum : array-like
        The background spectrum, in erg/s/cm^2/Ang/arcsec^2.
    '''
    # The wavelengths, in Angstroms
    lam = eclip_ilam[:, 0]
    # The specific intensity, in erg/s/cm^2/Ang/arcsec^2
    ilam = bkg_ilam(lam, eclip_angle)
    return np.array([lam, ilam])

def nLamberts_V_to_spec_radiance(nLamberts, lam=5500):
    '''Return a spectral radiance from a value in nLamberts measured in the V band.'''
    # First convert to Vmag/arcsec^2. Uses Eq. 1 in Krisciunas & Schaefer 1991.
    vmag = (20.7233 - np.log(nLamberts / 34.08)) / 0.92104
    # Convert to spectral radiance, in erg/s/cm^2/Hz/arcsec^2
    spec_radiance = 10 ** (-vmag / 2.5) * 3631 * 10 ** -23
    # Convert to erg/s/cm^2/Ang/arcsec^2
    spec_radiance = spec_radiance * (3 * 10 ** 18) / (lam ** 2)
    return spec_radiance


def moon_brightness_5500(alpha=180, rho=45, Zm=45, Zo=0, k=0.172):
    '''Return the brightness of the sky caused by moonglow. From Krisciunas & Schaefer 1991.

    Parameters
    ----------
    alpha : float
        The phase angle, in degrees. 0 is full moon, 90 is first quarter,
        180 is new moon. Should be between 0 and 180 (assumes symmetry
        between waning/waxing).
    rho : float
        The angular separation between the moon and the object, in degrees.
    k : float
        The atmospheric extinction coefficient, in mag/airmass.
    Zm : float
        The zenith angle of the moon, in degrees.
    Zo : float
        The zenith angle of the object, in degrees.
    Returns
    -------
    B_moon : float
        The spectral radiance of the sky caused by moonglow, in erg/s/cm^2/Ang/arcsec^2,
        at 5500 Angstroms.
    '''
    # Convert to radians
    rho_rad = np.radians(rho)
    Zm_rad = np.radians(Zm)
    Zo_rad = np.radians(Zo)
    # Intensity of the moon itself through the atmosphere
    Istar = 10 ** (-0.4 * (3.84 + 0.026 * alpha + 4e-9 * alpha ** 4))
    # Scattering function
    f = 10 ** 5.36 * (1.06 + np.cos(rho_rad) ** 2) + 10 ** (6.15 - rho / 40)
    Xm = (1 - 0.96 * np.sin(Zm_rad) ** 2) ** (-0.5)
    Xo = (1 - 0.96 * np.sin(Zo_rad) ** 2) ** (-0.5)
    # Eq. 15. Gives brightness in nLamberts
    B_moon = Istar * f * (10 ** (-0.4 * k * Xm)) * (1 - 10 ** (-0.4 * k * Xo))
    # Account for the opposition effect, whereby when near opposition, the moon
    # is brighter than expected due to specular reflections.
    if alpha < 7:
        B_moon *= 1 + (0.35 / 7) * (7 - alpha)
    # Convert from nLamberts to erg/s/cm^2/Ang/arcsec^2
    B_moon = nLamberts_V_to_spec_radiance(B_moon)
    return B_moon

def dark_sky_brightness_5500(Bzen=79, Zo=0, k=0.17):
    '''Brightness of the dark sky, as observed at Mauna Kea, in nLamberts.
    
    Parameters
    ----------
    Bzen : float
        The brightness of the sky at zenith, in nLamberts.
    Zo : float
        The zenith distace of the object, in degrees.
    k : float
        The atmospheric extinction coefficient, in mag/airmass.
    Returns
    -------
    Bdark : float
        The spectral radiance of the dark sky, in erg/s/cm^2/Ang/arcsec^2,
        at 5500 Angstroms.
    '''
    X = (1 - 0.96 * np.sin(np.radians(Zo)) ** 2) ** (-0.5)
    # Eq. 2 in Krisciunas & Schaefer 1991.
    B_dark = Bzen * 10 ** (-0.4 * k * (X-1)) * X
    B_dark = nLamberts_V_to_spec_radiance(B_dark)
    return B_dark

# Values for the dark sky brightness and brightness from the moon are for V band, centered
# at about 550nm. For other wavelengths, we'll scale these values based on the spectrum of
# moonlight, starlight, zodiacal light, and the airglow continuum measured at Paranal, using 
# the document "The Cerro Paranal Advanced Sky Model." These spectra are in Fig. 23 in that
# document, which we digitized in Paranal_Sky_Background_Spectrum.csv. This spectrum is
# also used to weight the contributions of scattered starlight, zodiacal light, and
# airglow continuum. We don't consider airglow lines.

bkg_components_spectra = np.genfromtxt(os.path.join(abs_path, '../data/Paranal_Sky_Background_Spectrum.csv'),
                                        delimiter=',')
wavelengths = bkg_components_spectra[:, 0]
moon_spectrum = bkg_components_spectra[:, 1]
dark_sky_total_spectrum = (bkg_components_spectra[:, 2] + bkg_components_spectra[:, 3] +
                           bkg_components_spectra[:, 4])

def bkg_spectral_radiance_ground(lam=5500, alpha=180, rho=45, Zm=45, Zo=0,
                                 Bzen=79, k=0.172):
    '''Return the spectral radiance of the sky background light at a given wavelength.'''
    moon_radiance_5500 = moon_brightness_5500(alpha, rho, Zm, Zo, k)
    dark_radiance_5500 = dark_sky_brightness_5500(Bzen, Zo, k)
    rel_moon_radiance = np.interp(lam, wavelengths, moon_spectrum) / np.interp(5500, wavelengths, moon_spectrum)
    rel_dark_radiance = np.interp(lam, wavelengths, dark_sky_total_spectrum) / np.interp(5500, wavelengths, dark_sky_total_spectrum)
    total_radiance = (moon_radiance_5500 * rel_moon_radiance +
                      dark_radiance_5500 * rel_dark_radiance)
    return total_radiance

def bkg_spectrum_ground(band=wavelengths, alpha=180, rho=45, Zm=45, Zo=0,
                        Bzen=79, k=0.172):
    '''Return the spectrum of light from the sky background across a given band'''
    # The wavelengths, in Angstroms
    lam = band
    # The specific intensity, in erg/s/cm^2/Ang/arcsec^2
    ilam = bkg_spectral_radiance_ground(lam, alpha, rho, Zm, Zo, Bzen, k)
    return np.array([lam, ilam])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    bkg_spectrum_ground_full = bkg_spectrum_ground(alpha=0)
    bkg_spectrum_ground_new = bkg_spectrum_ground(alpha=180)
    bkg_spectrum_space = bkg_spectrum_space(eclip_angle=90)
    plt.plot(bkg_spectrum_ground_full[0], bkg_spectrum_ground_full[1], 'o', label='Ground Observatory, Full Moon')
    plt.plot(bkg_spectrum_ground_new[0], bkg_spectrum_ground_new[1], 'o', label='Ground Observatory, New Moon')
    plt.plot(bkg_spectrum_space[0], bkg_spectrum_space[1], 'o', label='Space Observatory')
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Spectral Radiance (erg/s/cm^2/Ang/arcsec^2)')
    plt.legend()
    plt.yscale('log')
    plt.show()

    # downloads_folder = os.path.expanduser('~/Downloads')
    # moonlight2 = np.genfromtxt(os.path.join(downloads_folder, 'Moonlight2.csv'), delimiter=',')
    # continuum2 = np.genfromtxt(os.path.join(downloads_folder, 'Continuum2.csv'), delimiter=',')
    # starlight2 = np.genfromtxt(os.path.join(downloads_folder, 'Starlight2.csv'), delimiter=',')
    # zodiacal_light2 = np.genfromtxt(os.path.join(downloads_folder, 'ZodiacalLight2.csv'), delimiter=',')
    # continuum_new = np.interp(moonlight2[:, 0], continuum2[:, 0], continuum2[:, 1])
    # starlight_new = np.interp(moonlight2[:, 0], starlight2[:, 0], starlight2[:, 1])
    # zodiacal_light_new = np.interp(moonlight2[:, 0], zodiacal_light2[:, 0], zodiacal_light2[:, 1])
    # # Convert from dex
    # moonlight_new = 10 ** moonlight2[:, 1]
    # continuum_new = 10 ** continuum_new
    # starlight_new = 10 ** starlight_new
    # zodiacal_light_new = 10 ** zodiacal_light_new
    # # Save all 4 spectra to csv Paranal_Sky_Background_Spectrum.csv
    # np.savetxt(os.path.join(downloads_folder, 'Paranal_Sky_Background_Spectrum.csv'),
    #            np.array([moonlight2[:, 0]*10, moonlight_new, continuum_new,
    #                      starlight_new, zodiacal_light_new]).T,
    #            delimiter=',', header='Wavelength (Angstroms), Moonlight, Continuum, Starlight, Zodiacal Light')


