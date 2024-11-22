import math
import warnings
import numpy as np
import sigpy as sp
from scipy import signal
import scipy.io
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
#from scipy.integrate import cumtrapz
from scipy import integrate

import pypulseq as mr

# Frank's utils
#import utils # several helper functions for simulation and recon
#import MRzeroCore as mr0
from utils import animate, simulate_2d, reconstruct

# Set high-level sequence parameters
# Define FOV, resolution and other parameters
fov = 220e-3
Nx = 64 #60
Ny = 64 #60
slice_thickness = 4e-3  # Slice thickness
n_slices = 1
ro_duration=520e-6 #1200e-6 # duration of the ADC / readout event, defailt: 1200us, 240us still works
rf_duration=2.5e-3

# Set system limits
system = mr.Opts(
    max_grad=32,
    grad_unit="mT/m",
    max_slew=130,
    slew_unit="T/m/s",
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
)

# More advanced options and switches. you may chose to disable some of these options to save time in the exercises
plot: bool = True
animate_seq: bool = True
simulate: bool = True
write_seq: bool = True
seq_filename: str = "epi_pypulseq.seq"


"""# Create Pulseq objects"""
# Create 90 degree slice selection pulse and gradient
rf, gz, gz_reph = mr.make_sinc_pulse(
    flip_angle=np.pi / 2,
    system=system,
    duration=rf_duration,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    return_gz=True,
)

# Define other gradients and ADC events
delta_k = 1 / fov
k_width = Nx * delta_k
print('delta_k',delta_k)
print('k_width',k_width)
dwell_time = ro_duration / Nx
flat_time = np.ceil(ro_duration / system.grad_raster_time) * system.grad_raster_time  # round-up to the gradient raster
gx = mr.make_trapezoid(
    channel="x",
    system=system,
    amplitude=k_width / ro_duration,
    flat_time=flat_time,
)



gx_even = mr.make_trapezoid(
    channel="x",
    system=system,
    #amplitude = (k_width + 0.05) / ro_duration,
    amplitude = (k_width) / ro_duration,
    flat_time=flat_time,
)



gx_odd = mr.make_trapezoid(
    channel="x",
    system=system,
    #amplitude = (k_width) / ro_duration - 0.127,
    amplitude = -(k_width) / ro_duration,
    flat_time=flat_time,
)

adc = mr.make_adc(
    num_samples=Nx,
    duration=ro_duration,
    delay=gx.rise_time + flat_time / 2 - (dwell_time*Nx) / 2,
    system=system,
)

# Pre-phasing gradients (in the shortest possible time or filling the available time)
gx_pre = mr.make_trapezoid(
    channel="x", system=system, area=-gx.area / 2, duration=mr.calc_duration(gz_reph),
)
gy_pre = mr.make_trapezoid(
    channel="y", system=system, area=(Ny / 2 - 1) * delta_k, duration=mr.calc_duration(gz_reph),
)

# align gradients (calculate their delays)
gx_pre, gy_pre, gz_reph=mr.align(right=[gx_pre],left=[gy_pre, gz_reph])

# Phase blip in the shortest possible time (assuming a triangle is possible)
dur = np.ceil(2 * np.sqrt(delta_k / system.max_slew) / system.grad_raster_time) * system.grad_raster_time
gy = mr.make_trapezoid(channel="y", system=system, area=-delta_k, duration=dur)

print('achieved echo spacing is {}us'.format(round(1e6*(mr.calc_duration(gx)+mr.calc_duration(gy)))))


#""" # Construct the sequence (seq) and a line by line modified Sequence (seq2) """
seq2 = mr.Sequence(system)  # Create a new sequence object
Ny_2 = int(Ny/2)
measured_offset_c_even = np.load('diff_center_kx_p_even.npy')
measured_offset_c_odd = np.load('diff_center_kx_p_odd.npy')
n_measurements = measured_offset_c_even.shape[0]
print('n_measurements', n_measurements)
# populate the sequence object with event blocks
seq_list = []
#n_measurements = 2
for m in range(n_measurements):
  seq2 = mr.Sequence(system)  # Create a new sequence object
  for s in range(n_slices):
    rf.freq_offset = gz.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
    seq2.add_block(rf, gz)
    seq2.add_block(gx_pre, gy_pre, gz_reph)

    for i in range(Ny_2):
      #print('measured_offset_c_even[i,1]',measured_offset_c_even[1,i])
      #print('measured_offset_c_odd[i,1]',measured_offset_c_odd[i,1])
      gx_even = mr.make_trapezoid(channel="x", system=system,
        amplitude = (k_width + measured_offset_c_even[m,i]*2) / ro_duration,
        flat_time=flat_time)
      gx_odd = mr.make_trapezoid(channel="x", system=system,
        amplitude = -(k_width - measured_offset_c_odd[m,i]*2) / ro_duration,
        flat_time=flat_time)

      seq2.add_block(gx_even, adc)  #
      seq2.add_block(gy)  # Phase blip
      seq2.add_block(gx_odd, adc)
      if i!=Ny_2-1:
        seq2.add_block(gy)  # Phase blip
        #gx_curr = mr.scale_grad(gx_curr,-1)  # Reverse polarity of read gradient
  seq_list.append(seq2)

print('len(seq_list)',len(seq_list))
#namespaceobject

seq = mr.Sequence(system)  # Create a new sequence object
Ny_2 = int(Ny/2)

# populate the sequence object with event blocks
for s in range(n_slices):
  rf.freq_offset = gz.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
  seq.add_block(rf, gz)
  seq.add_block(gx_pre, gy_pre, gz_reph)
  gx_curr=gx

  for i in range(Ny):
    seq.add_block(gx_curr, adc)  # Read one line of k-space out
    if i!=Ny-1:
      seq.add_block(gy)  # Phase blip
      gx_curr = mr.scale_grad(gx_curr,-1)  # Reverse polarity of read gradient



""" import measured trajektories """
[k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc] = seq_list[1].calculate_kspace()
kdata_measured_cimax = np.load('kData_cimaX.npy')/(2*np.pi)
kdata_measured_prisma = np.load('kData_prisma.npy')/(2*np.pi)
kdata_measured_cimax = kdata_measured_prisma #quick ugly hack preISMRM
kstart = 0
kend = 5119
kstart = 5119
#kstart = 5119 + 25
kend = 37599 + 4640 - 14 
#kend = 37599 + 4640 - 14 - 28
#kend = 43119


#fig, ax = plt.subplots()
kx_d = signal.detrend(kdata_measured_cimax[kstart:kend,1,0,0])
kx = (kdata_measured_cimax[kstart:kend,1,0,0])
ky = kdata_measured_cimax[kstart:kend,2,0,0]
ky_d = signal.detrend(kdata_measured_cimax[kstart:kend,2,0,0].T,axis=0).T

#k0 = kdata_measured_cimax[kstart:kend,0,0,0]
k0 = kdata_measured_cimax[kstart:kend,0,0,:]
print(t_adc[1]-t_adc[0],'t_adc[1]-t_adc[0]')
tk_start = 0.00010
tk_end = 0.00020
k_time_slice = np.linspace(t_adc[0]-tk_start, t_adc[-1]+tk_end, k0.shape[0]) #wrong this is k_time_slice
print('k_time_slice.shape',k_time_slice.shape)
k_time_seq = np.ones((k0.shape[1], 1)) * k_time_slice
k_time_seq = k_time_seq.T
print('k_time_seq.shape',k_time_seq.shape)

#plt.figure(figsize=(10, 10))
#plt.plot(k_time_seq, k0, 'k.')
#plt.plot(t_adc, k0_t_adc, 'r.')
#plt.title('k-space trajectory')
#plt.xlabel('time [s]')
#plt.ylabel(r'$k_0 \mathregular{\ /m^{-1}}$')
#plt.show()

#plt.plot(t_adc, 'r.')
#plt.plot(kx,ky, 'k-')
#ax.set_aspect('equal', adjustable='box')

# Sum k0 up until each t_adc point
#k0_t_adc_cumsum = np.cumsum(k0_t_adc)  # Cumulative sum of k0_t_adc
print('k0[:,0].shape',k0[:,0].shape)

""" Detrend k0 on a slice by slice basis, to get rid of static B0 (bad shim) effects"""
#k0 = signal.detrend(k0,axis=0)
""" Detrend k0 on an all experiments basis to get rid of static B0 effects ?? correct??"""
k0 = signal.detrend(signal.detrend(k0.T).T)


#wrap k0

""" Integrating k0 to figure out the accumulated k0 offset ??? correct?? """
t_adc_center = t_adc[32:-1:64]
#k0 = integrate.cumulative_trapezoid(k0, k_time_seq , axis=0, initial=0)
#ynew = np.interp(xnew, x, y)
#interpolator_trapz = np.interp(k_time_seq, t_adc, k0_cumtrapz)
#interpolator_trapz = interp1d(k_time_seq, k0_cumtrapz, kind='linear', fill_value="extrapolate", axis = 1)
#interpolator = interp1d(k_time_seq, k0, kind='linear', fill_value="extrapolate")

#t_adc_center = t_adc[32:-1:64]
##k0_t_adc = np.interp(t_adc, k_time_seq[:,0], k0[:,0])
#k0_t_adc = np.interp(t_adc, k_time_seq[:,0], k0[:,0])
#k0_t_adc_center = np.interp(t_adc_center, k_time_seq[:,0], k0[:,0])
#
##interpolator = interp1d(k_time_seq[:,0], k0[:,0], kind='linear', fill_value="extrapolate")
##interpolator = interp1d(k_time_seq, k0[:,0], kind='linear', fill_value="extrapolate")
##interpolator_2 = interp1d(k_time_seq, k0[:,1], kind='linear', fill_value="extrapolate")
##print('t_adc_center',t_adc_center)
###k0_t_adc = interpolator(t_adc)  # Interpolated values at t_adc points
##k0_t_adc_2 = interpolator_2(t_adc)  # Interpolated values at t_adc points
###k0_cumsum_t_adc = interpolator(t_adc)  # Interpolated values at t_adc points
##print('k0_t_adc.shape',k0_t_adc.shape)
##k0_t_adc_center = interpolator(t_adc_center)  # Interpolated values at t_adc points
###k0_cumsum_t_adc_center = interpolator(t_adc_center)  # Interpolated values at t_adc points
#
#k0_mean_t_adc_center_one = np.mean(k0_t_adc[0:64])# better use k0_cumsum not k0_cumsum_t_adc
##k0_mean_t_adc_center_one = np.mean(k0_cumsum_t_adc[0:64])# better use k0_cumsum not k0_cumsum_t_adc
#print('k0_mean_t_adc_center_one',k0_mean_t_adc_center_one)
#k0_t_r axis== k0_t_adc.reshape(64,64)
##k0_c_t_r = k0_cumsum_t_adc.reshape(64,64)
#print('k0_c_t_r.shape', k0_t_r.shape )
#k0_mean_t_adc_center = np.mean(k0_t_r, axis=1) 
#print('k0_mean_t_adc_center',k0_mean_t_adc_center.shape)


""" Interpolate k0 to t_adc """
# Interpolate each column in k0 using k_time_seq for all columns
k0_t_adc = np.array([np.interp(t_adc, k_time_seq[:, i], k0[:, i]) for i in range(k0.shape[1])]).T
k0_t_adc_center = np.array([np.interp(t_adc_center, k_time_seq[:, i], k0[:, i]) for i in range(k0.shape[1])]).T
print('k0_t_adc.shape',k0_t_adc.shape)


# Plotting
plt.figure(figsize=(10, 10))
plt.plot(k_time_seq[:,0], k0[:,0], 'k.', label="Original k0 data")
#plt.plot(k_time_seq[:,0], k0[:,0], 'k.', label="Original k0 data")
plt.plot(t_adc, k0_t_adc[:,0], 'r.', label="Interpolated k0 data at t_adc")
#plt.plot(t_adc, k0_t_adc[:,99], 'g.', label="Interpolated k0 data at t_adc")
#plt.plot(t_adc, k0_t_adc[:,0], 'r.', label="Interpolated k0 data at t_adc")
#plt.plot(t_adc, k0_cumsum_t_adc, 'r.', label="Interpolated k0 data at t_adc")
#plt.plot(t_adc_center, k0_t_adc_center, 'b.', label="Interpolated k0 data at t_adc")
#plt.plot(t_adc_center, k0_cumsum_t_adc_center, 'b.', label="Interpolated k0 data at t_adc")
#plt.plot(t_adc_center, k0_mean_t_adc_center, 'g.', label="Interpolated k0 data at t_adc")
plt.title('k-space trajectory')
plt.xlabel('time [s]')
plt.ylabel(r'$k_0 \mathregular{\ /m^{-1}}$')
plt.legend()
#plt.show()




""" # Visualize the sequence """
#print(k_traj_adc[0],'k_traj_adc')
#print(k_traj_adc[1,],'k_traj_adc1')
print('t_adc.shape',t_adc.shape)
length_t_adc = t_adc[-1]-t_adc[0]
print('length_t_adc',length_t_adc)
print('t_excitation',t_excitation)
print('k0.shape',k0.shape)
plt.figure(figsize=(10, 10))
#plt.plot(k_traj[0,],k_traj[1,], 'b-')
#plt.plot(k_traj_adc[0,],k_traj_adc[1,], 'r.')
###plt.plot(kdata_measured_cimax[0,],kdata_measured_cimax[1,], 'k-')
plt.plot(kx,ky, 'k.')
plt.plot(kx_d,ky, 'r.')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.title('k-space trajectory')
plt.xlabel(r'$k_x \mathregular{\ /m^{-1}}$')
plt.ylabel(r'$k_y \mathregular{\ /m^{-1}}$')
#plt.draw()
plt.show()





""" # (Optionally) Analyze the sequence """
rep = seq.test_report()
#print(rep)

""" Prepare the sequence output for the scanner """
seq.set_definition(key="FOV", value=[fov, fov, slice_thickness*n_slices])
seq.set_definition(key="Name", value="epi")

#if write_seq:
#  # Write the sequence file
#  seq.write(seq_filename)
##  from google.colab import files
##  files.download(seq_filename)  # Download locally

""" load cropped_phantom.mat """
sim_filename = 'cropped_phantom.mat'

""" Simulate the sequence """
#if simulate:
#  # Simulate sequence
#  #kdata = mr0.compute_graph(seq, obj_p, 200, 1e-3)
#  #kdata2 = mr0.compute_graph(seq2, obj_p, 200, 1e-3)
#
#  kdata = simulate_2d(seq, sim_size=[Nx*2,Ny*2], noise_level=0, n_coils=1) # B0_polynomial=[...] optional parameter: B0_offset, Gx, Gy, x^2, y^2, x*y; units??? good value 200
#  kdata2 = simulate_2d(seq2, sim_size=[Nx*2,Ny*2], noise_level=0, n_coils=1)
#  print('kdata2.shape',kdata2.shape)
#  print('kdata2',kdata2)
#  print('kdata2.dtype',kdata2.dtype)
#
#  kdata_list = np.zeros((1,64,64,n_measurements),dtype=np.complex64)
#  for m in range(n_measurements):
#      print('m',m)
#      #for m in range(1): #for testing
#      kdata_list[:,:,:,m] = simulate_2d(seq_list[m], sim_size=[Nx*2,Ny*2], noise_level=0, n_coils=1)
#
#
#  # Display raw data profiles
#  plt.figure()
#  plt.imshow(abs(kdata[0].reshape(-1,adc.num_samples)))
#  plt.title('Magnitude of acquired raw data'); # raw data, not k-space data (odd/even lines not reverted)
#
#  # Display the data more correctly
#  kdata_swapped=np.zeros_like(kdata)
#  kdata_swapped[:,0::2,:]=kdata[:,0::2,:]
#  kdata_swapped[:,1::2,:]=kdata[:,1::2,::-1]
#  plt.figure()
#  plt.imshow(abs(kdata_swapped[0].reshape(-1,adc.num_samples)))
#  plt.title('Magnitude of k-space data (odd lines swapped)'); # raw data, not k-space data (odd/even lines not reverted)
#
#
#""" save kdata and kdata2 """
#
#np.save('kdata_p.npy',kdata)
#np.save('kdata2_p.npy',kdata2)
#np.save('kdata_list_p.npy',kdata_list)

""" load kdata and kdata2 """
kdata = np.load('kdata_p.npy')
#kdata2 = np.load('kdata2_p.npy')
kdata2 = np.load('kdata_p.npy')
kdata_list = np.load('kdata_list_p.npy')

#""" create kdata2_list by repetition of kdata2 to test the k0 modification """
#kdata2_list = np.zeros((1,64,64,n_measurements),dtype=np.complex64)
#for m in range(n_measurements):
#    kdata2_list[:,:,:,m] = kdata2
#
#kdata_list = kdata2_list


#print('kdata.shape',kdata.shape)
#print('kdata2',kdata2[:,:,0])
#print('kdata2.shape',kdata2.shape)

""" k_line B0 modification test"""
#print(t_adc,'t_adc')
#deltaB0_phase_per_second = np.deg2rad(19000)
magnitude = np.abs(kdata2)
phase = np.angle(kdata2)
print('magnitude.shape',magnitude.shape)
phase_mod_t_adc = k0_t_adc[:,0] #
print('phase_mod_t_adc.shape',phase_mod_t_adc.shape)
#phase_mod_t_adc = k0_cumsum_t_adc #/100
phase_mod = phase_mod_t_adc.reshape(1, 64, 64)
print('phase_mod.shape',phase_mod.shape)
phase_new = phase + phase_mod

kdata2 = magnitude * np.exp(1j * phase_new)
kdata_dif = kdata - kdata2 
print('kdata_dif',kdata_dif)

""" k_line B0 modification of kdata_list"""
print('kdata_list.shape',kdata_list.shape)
magnitude = np.abs(kdata_list) #?axis
phase = np.angle(kdata_list) #?axis
print('phase.shape',phase.shape)

phase_mod_t_adc = k0_t_adc #/100
print('phase_mod_t_adc.shape',phase_mod_t_adc.shape)
phase_mod_test = phase_mod_t_adc.reshape(1, 64, 64, 100)
print('phase_mod.shape',phase_mod.shape)
print('phase_test0', phase_mod)
print('phase_test', phase_mod_test[:,:,:,0]-phase_mod)
print('phase_test2', phase_mod_test[:,:,:,99]-phase_mod)
phase_mod = phase_mod_t_adc.reshape(1, 64, 64, 100)
phase_new = phase + phase_mod

kdata_list = magnitude * np.exp(1j * phase_new)
print('kdata_list.shape',kdata_list.shape)

kdata_list_dif = kdata_list[:,:,:,99] - kdata2 
print('kdata_list_dif',kdata_list_dif)

""" Reconstruct the image """
if simulate:

  rec = reconstruct(kdata, seq, trajectory_delay=0e-6, cartesian=False) # for the trajectory delay to work with the basic (non-ramp-sampling EPI) cartesian=False is needed
  # need to invert data dimentions because the reconstruction code uses iFFT to go from k-space to image space
  rec=rec[::-1,::-1]

  rec2 = reconstruct(kdata2, seq, trajectory_delay=0e-6, cartesian=False) # for the trajectory delay to work with the basic (non-ramp-sampling EPI) cartesian=False is needed
  # need to invert data dimentions because the reconstruction code uses iFFT to go from k-space to image space
  rec2=rec2[::-1,::-1]
  print('rec2.shape',rec2.shape)
  #print('rec2',rec2)
  print('rec2.dtype',rec2.dtype)

  rec_list = np.zeros((66,64,n_measurements))
  for m in range(n_measurements):
      print('m',m)
  #for m in range(1): #for testing
      #kdata_temp = kdata2_list[:,:,:,m]
      kdata_temp = kdata_list[:,:,:,m]
      #print('kdata_temp.shape',kdata_temp.shape)
      #rec_temp = reconstruct(kdata2, seq, trajectory_delay=0e-6, cartesian=False)
      rec_temp = reconstruct(kdata_temp, seq, trajectory_delay=0e-6, cartesian=False)
      rec_list[:,:,m] = rec_temp[::-1,::-1]
      #rec_list.append(rec_temp)

  #rec_list = np.load('recon_list_CimaX.npy')

  fig, ax = plt.subplots(nrows=1, ncols=3)
  ax = ax.flatten()

  ax[0].imshow(abs(rec), origin='lower')
  #ax[0].imshow(abs(rec_list[:,:,10]), origin='lower')
  ax[0].set_title('kdata')

  #ax[1].imshow(abs(rec_list[:,:,61]), origin='lower')
  ax[1].imshow(abs(rec2), origin='lower')
  ax[1].set_title('kdata2')
  print('test')
  ax[2].imshow(abs(rec2)-abs(rec), origin='lower')
  #ax[2].imshow(abs(rec_list[:,:,10])-abs(rec_list[:,:,61]), origin='lower')
  ax[2].set_title('kdata3')

  fig.tight_layout()
  fig.set_figheight(6)
  fig.set_figwidth(18)
  plt.show();

"""save"""
#np.save('recon_list_CimaX.npy',rec_list)
np.save('recon_list_Prisma.npy',rec_list)
""" save as matlab file """
#scipy.io.savemat('recon_list_CimaX.mat', mdict={'rec_list': rec_list})
scipy.io.savemat('recon_list_Prisma.mat', mdict={'rec_list': rec_list})


