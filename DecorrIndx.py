import scipy
import numpy as np
import pandas as pd
import os, glob, datetime
import matplotlib.cm as cm
from obspy.core import read
import matplotlib.pyplot as plt
from obspy.core.trace import Trace
from matplotlib.dates import date2num
from obspy.signal.filter import bandpass
from obspy.signal.invsim import cosine_taper
from obspy.io.sac.util import get_sac_reftime
from obspy.signal.cross_correlation import xcorr_pick_correction, correlate, xcorr_max

from numba import jit
import timeit



def decorr_index(reference, stream_waves, trace_len, bandpass_filter, windowing):

	'''
	Assuming is the same station we're going to measure the delay in time between 
	two repeating events using a moving window.  

	Template is an obspy stream with the waveform template
	stream_waves is an obspy stream with at least one waveform
	'''

	def time_Ppicks(trace):

		ref_time_event = get_sac_reftime(trace.stats.sac)
		ptime_pick = ref_time_event + trace.stats.sac.a

		return ptime_pick

	
	def prep_data(waveform):
		st = read(waveform)
		tr = st[0]

		# Creating a time_vector in seconds
		sampling_rate = tr.stats.sampling_rate
		delta = 1.0/sampling_rate

		# getting the p-wave pick time
		ptime_pick = time_Ppicks(tr)

		# Removing mean and trend for template
		tr.detrend(type="demean")
		tr.detrend('linear')

		L = len(tr.data)
		time = np.arange(0, L)*delta

		return tr, ptime_pick

	@jit
	def rolling_window(a, b, wlen, stp):

		Decorr = []
		Corr =[]
		for kwin, window in enumerate(a.slide(window_length=wlen, step=stp)):
			tmpa = window.copy()
			time = np.arange(0, len(tmpa))*tmpa.stats.delta/100
			stime = tmpa.stats.starttime
			etime = tmpa.stats.endtime
			tmpb = b.slice(starttime=stime, endtime=etime, nearest_sample=True)

			# Cross-correlation between a_i and b_j windows in the freq. domain: 
			a_i = tmpa.data
			b_i = tmpb.data

			cc = correlate(a_i, b_i, normalize=True, domain='time', shift=2, demean=True)
			shift, value = xcorr_max(cc)
			value = np.around(value, 3)

			if value < 0.0:
				value = 0.0
			else: 
				value = value

			dec_inx = 1.0 - value
			cc_inx = value

			Decorr.append(dec_inx)
			Corr.append(cc_inx)

		# It returns a list with the decorr index and cc index as a function of time. 
		return Decorr, Corr


	@jit
	def moving_window(reference, stream_waves, t_before_p, t_after_p, lp, hp, windowing):

		target, p1 = prep_data(reference)
		
		DECORR = []
		CORR = []
		STIME = []
		Traces_good_cc = []
		Targets=[]

		stream_waves.sort()

		for ist in stream_waves:

			if not ist == reference:

				ij_trace, ij_pick = prep_data(ist)
				## Performing a cross-correlation for aligning the two seismograms at 
				## the maximum cross-correlation coefficient.
				lag_time, coeff = xcorr_pick_correction(p1, target, ij_pick, ij_trace, t_before=0.05,
														t_after=4.0, cc_maxlag=1.0, filter="bandpass", 
														filter_options={'freqmin': lp, 'freqmax': hp},
														plot=False,)
				# coeff >= 0.9699
				if coeff >= 0.90:

					STIME.append(ij_trace.stats.starttime)
					print("Reference trace vs %s, CC = %s" %(ist, coeff))

					# Correcting both seismograms
					corrected_target = target.trim(p1 - (t_before_p), p1 + (t_after_p))
					corrected_event2 = ij_trace.trim(ij_pick - (t_before_p - lag_time), ij_pick + (t_after_p + lag_time))
					
					# Applying a cosine taper
					corrected_target.data *= cosine_taper(len(corrected_target), 0.1)
					corrected_event2.data *= cosine_taper(len(corrected_event2), 0.1)

					# Sampling rate for the reference and the second waveform
					sp_t = corrected_target.stats.sampling_rate
					st_ev2 = corrected_event2.stats.sampling_rate

					# Now, we need to filter the traces before measuring de-correlation index
					corr_tar_filt =  Trace(bandpass(corrected_target, lp, hp, sp_t, corners=4, zerophase=True))
					corr_tar2_filt = Trace(bandpass(corrected_event2, lp, hp, st_ev2, corners=4, zerophase=True))

					# Interpolating to 1000 Hz to make an smooth measurement
					corr_tar_filt_int = corr_tar_filt.resample(sampling_rate=1000, )
					corr_tar2_filt_int = corr_tar2_filt.resample(sampling_rate=1000,)

					# Checking the lenght of the waveforms. They must correspond
					l1, l2 = len(corr_tar_filt), len(corr_tar2_filt)

					if l1 != l2:
						_msg = "Waveforms have different window_lenght, check data = " + ist
						raise IOError(_msg)

					else:
						a = corr_tar_filt_int
						b = corr_tar2_filt_int

					# Before sliding, let's create a copy of the traces
					a_c = a.copy()
					b_c = b.copy()

					Traces_good_cc.append(b_c)
					Targets.append(a_c)

					# Window lenght and step function for the moving window. 
					wl = windowing[0] * 100
					stp = windowing[1] * 100
					decorr, corr = rolling_window(a, b, wlen=wl, stp=stp)

					DECORR.append(decorr)
					CORR.append(corr)

		return DECORR, CORR, STIME, Traces_good_cc, Targets

		
	lp=bandpass_filter[0]
	hp=bandpass_filter[1]

	DECORR, CORR, STIME, Traces_good_cc, Targets = moving_window(reference=reference, stream_waves=stream_waves, 
			   	   												t_before_p=trace_len[0], 
									   	   						t_after_p=trace_len[1],
									   	   						lp=lp, hp=hp, 
									   	   						windowing=windowing)
	def get_colors(inp, colormap, vmin, vmax):
	    norm= plt.Normalize(vmin, vmax)
	    return colormap(norm(inp))

	fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(12, 9.5))

	sps=1000
	delta = 1/sps

	K = np.arange(0, len(Traces_good_cc))

	L = [2*k+2 for k in range(len(Traces_good_cc))]
	L = L[::-1]
	max_L = np.max(L) + 1.7

	m = cm.ScalarMappable(cmap=cm.winter)
	m.set_array(L)
	colors = get_colors(L, plt.cm.winter, vmin=np.min(L), vmax=np.max(L))

	wave_time_vec = np.arange(0, len(Traces_good_cc[0]))*delta

	for kk in K:
		tr = Traces_good_cc[kk].normalize()
		data = tr.data
		ax[0].plot(wave_time_vec, data+L[kk], color=colors[kk], lw=1)

	ax[0].get_xaxis().set_visible(False)
	ax[0].get_yaxis().set_visible(False)
	ax[0].set_xlim(0, np.max(wave_time_vec))
	ax[0].set_ylabel("Normalized amplitude")
	ax[0].title.set_text('Station OCM, HHZ. bp = %s - %s Hz' %(bandpass_filter[0], bandpass_filter[1]))

	winvec = np.arange(0, len(DECORR[0]))*windowing[1]
	for d, dec in enumerate(DECORR):
		ax[1].plot(winvec, dec, color=colors[d], label=STIME[d])

	ax[1].set_xlabel("Time, s")
	ax[1].set_ylabel("Decorrelation index")
	ax[1].set_ylim(0, 1.0)
	ax[1].set_xlim(0, np.max(winvec))
	ax[1].legend()


	plt.show()
	fig_name = "Decorrelation_Index_Bp_" + str(bandpass_filter[0]) + "_" + str(bandpass_filter[1]) + ".png"
	fig.savefig(fig_name, format='png', dpi=700)



# ----------------I N P U T   P A R A M S----------------------- ###

bandpass_filter=[2, 20]
trace_len=[0.6, 10.4]
windowing = [1, 0.10]

# ----------------D A T A   L I S T S----------------------- ###
reference = 'Highly_rep/OCM.HHZ.2019.108.20.27.15'
data_stream = glob.glob('Highly_rep/*HH*')

# ----------------R U N  T H E  C O D E----------------------- ###
decorr_index(reference=reference, stream_waves=data_stream, 
			 trace_len=trace_len, bandpass_filter=bandpass_filter, 
			 windowing=windowing)




