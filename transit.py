import batman
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt

target = "Kepler-11"
low = 1.0
high = 120.0

# Setup model light curve
params = batman.TransitParams()       #object to store transit parameters
params.t0 = 0.                        #time of inferior conjunction
params.per = 1.                       #orbital period
params.rp = 0.0285                    #planet radius (in units of stellar radii)
params.a = 33.330                     #semi-major axis (in units of stellar radii)
params.inc = 90.0                     #orbital inclination (in degrees)
params.ecc = 0.004                    #eccentricity
params.w = 90.                        #longitude of periastron (in degrees)
params.limb_dark = "nonlinear"        #limb darkening model
params.u = [0.5, 0.1, 0.1, -0.1]      #limb darkening coefficients [u1, u2, u3, u4]

t = np.linspace(-0.025, 0.025, 1000)
m = batman.TransitModel(params, t)
flux = m.light_curve(params)

def corrector_func(lcc):
    corrected_lc = lcc.normalize().remove_nans().remove_outliers().fill_gaps().flatten(window_length=401, sigma=6)
    return corrected_lc

# Download data
print("Downloading %s pixel file ..." % target)
lk.search_targetpixelfile("KEPLER-11", quarter=1).download(quality_bitmask='hardest').plot(frame=1)

print("Downloading %s light curves ..." % target)
lc = lk.search_lightcurvefile(target, mission='Kepler').download_all().PDCSAP_FLUX.stitch(corrector_func=corrector_func)
lc.scatter()
plt.show()

# Find transit
print('\nCreating periodograms ...')

pg_first = lc.to_periodogram(method="bls", period=np.arange(low, high, 0.001))
pg_first.plot(label="%s Transit Periodogram" % target)

# Find for Second eclipse
p = pg_first.period_at_max_power.value
pg_second = lc.to_periodogram(method="bls", period=np.arange((p / 2.0) - 1.0, (p / 2.0) + 1.0, 0.00001))
pg_second.plot(label="%s Second Eclipse Periodogram" % target)

# Create plots
print('Creating plots ...')
lc_first = lc.fold(period=pg_first.period_at_max_power, t0=pg_first.transit_time_at_max_power)

ax = lc_first.scatter(label="Flux")
plt.xlim(-0.025, 0.025)

lc_first.errorbar(ax=ax, label="Error bar")
plt.xlim(-0.025, 0.025)

plt.plot(t, flux, label='Model')
plt.legend()
plt.show()

# Second eclipse
lc_second = lc.fold(period=pg_second.period_at_max_power, t0=pg_second.transit_time_at_max_power)
lc_second.bin(binsize=10).scatter(label="%s Second Eclipse" % target)

ax2 = lc_second.scatter(label="Flux")
plt.xlim(-0.025, 0.025)

lc_second.errorbar(ax=ax2, label="Error bar")
plt.xlim(-0.025, 0.025)

print("\nTransit CCPD noise: %f" % lc_first.estimate_cdpp())
print("Second eclipse CCPD noise: %f" % lc_second.estimate_cdpp())
print('Strongest transit Period: %f %s' % (pg_first.period_at_max_power.value, pg_first.period_at_max_power.unit))
print('Strongest transit t0: %f' % pg_first.transit_time_at_max_power)
print('Second eclipse Period: %f %s' % (pg_second.period_at_max_power.value, pg_second.period_at_max_power.unit))
print('Second eclipse offset: %f %s' % (pg_second.period_at_max_power.value - (pg_first.period_at_max_power.value / 2), pg_second.period_at_max_power.unit))
print('Transit minimum flux: %f' % np.amin(lc_first.bin(binsize=10).flux))
print('Second eclipse minimum flux: %f' % np.amin(lc_second.bin(binsize=10).flux))
print('Planet jupiter radii: %f RJ' % ((1.065 * np.math.sqrt(1.0 - np.amin(lc_first.bin(binsize=10, method="median").flux))) * 9.73116))
print('Phase ∝ %f d' % (pg_first.transit_time_at_max_power + (1.0 * pg_first.period_at_max_power.value)))
# t0 + n * period

plt.show()
