#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
import pdb

def symbol_timing_recovery(data, sps=4):
    val = -2.0*np.pi/float(sps)
    indices = np.arange(data.size)
    mag_squared = np.abs(data)
    mag_squared **= 2
    Xm = np.dot(mag_squared, np.exp(1j*val*indices))
    t_estimate = -0.5*np.angle(Xm)/np.pi
    print('Symbol timing correction: {}'.format(t_estimate))

    y = indices/float(sps);
    f = interpolate.interp1d(y, data, copy=False, bounds_error=False, fill_value=0.)
    xnew = f(y + t_estimate)
    return xnew


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot a binary data files generated from channelizer_client_api_test')
    parser.add_argument('file', help='full path to data file')
    parser.add_argument('samplerate', help='Sample rate of data in Hz')
    parser.add_argument('-t', '--type', help='data type. Either \'cfloat\', \'float\', \'cshort\', \'short\'. Default is cfloat')
    parser.add_argument('-l', '--length', help='the number of samples to plot')
    parser.add_argument('-s', '--start', help='the sample index (0-based) to start with. Default is 0.')
    parser.add_argument('-d', '--downsample', help='the number of samples to downsample by before plotting')
    parser.add_argument('-a', '--average-length', help='length of moving averaging filter to apply')
    parser.add_argument('-S', '--SPS', default=4, help='Oversample factor of data - default is 4')
    parser.add_argument('-c', '--constellation', action='store_true', default=False, help='Generate constellation plot. Assumes 4x Oversampled data.')
    args = parser.parse_args()

    exists = os.path.isfile(args.file)
    if exists:
        if args.type is None or args.type == 'cfloat':
            print 'Reading in as complex-float'
            x = np.fromfile(args.file, dtype=np.complex64)
            t = np.arange(0.0, x.size)
        elif args.type == 'cshort':
            print 'Reading in as complex-short'
            x = np.fromfile(args.file, dtype=np.int16)
            x = x[0::2] + 1j*x[1::2]
            x = x.astype(np.complex64)
            t = np.arange(0.0, x.size)
        elif args.type is None or args.type == 'float':
            print 'Reading in as float'
            x = np.fromfile(args.file, dtype=np.float32)
            x = x.astype(np.complex64)
            t = np.arange(0.0, x.size)
        elif args.type == 'short':
            print 'Reading in as short'
            x = np.fromfile(args.file, dtype=np.int16)
            x = x.astype(np.complex64)
            t = np.arange(0.0, x.size)
        else:
            print 'Unrecognized data type', args.type
    else:
        print 'Could not load file:', args.file
        exit(1)
        
    if args.constellation:
        print('Oversample rate is assumed to be 4')
    
    print 'Read in', x.size, 'samples'
    
    d = 1
    if args.downsample is not None and int(args.downsample)>0:
        d = int(args.downsample)
        print 'Downsampling by', d

    SPS = float(args.SPS)
    print ('Using SPS = {}'.format(SPS))

    L = x.size
    if args.length is not None:
        if L < args.length:
            L = int(args.length)
            print 'Trimming to length', L
            
    start = 0
    if args.start is not None:
        start = int(args.start)
        if start+L >= x.size:
            print 'Starting sample is too large by', (start+L) - x.size
            exit(1)
        print 'Starting sample is', start
    
    x = x[start:start+L:d]
    t = t[start:start+L:d]
    
    x_dB = 20*np.log10(np.abs(x), where=np.abs(x)>0)
    if args.average_length is not None and int(args.average_length)>1 and int(args.average_length)<len(x):
        h = np.ones((int(args.average_length),), dtype=x_dB.dtype) / float(args.average_length)
        x_dB = signal.fftconvolve(x_dB, h, mode='same')

    fs = float(args.samplerate)
    t = t/fs

    fig = plt.gcf()
    fig.suptitle(args.file, fontsize=12)

    ax1 = plt.subplot(3, 1, 1)
    plt.plot(t, x.real)
    plt.plot(t, x.imag)
    plt.grid(True)
    plt.ylabel('Re & Im (mV)')

    ax2 = plt.subplot(3, 1, 2)
    plt.plot(t, x_dB)
    plt.grid(True)
    plt.ylabel('Signal Power (dBm)')

    ax3 = plt.subplot(3, 1, 3)
    plt.psd(x, NFFT=256, Fs=fs, Fc=0.0)
    plt.grid(True)
    plt.ylabel('dBm')

    plt.show()
    
    if args.constellation:
        # Try to correct for timing errors
        x = symbol_timing_recovery(x, sps=SPS)
        fig, axs = plt.subplots(1, 1)
        plt.scatter(x.real[::SPS], x.imag[::SPS], alpha=0.25, edgecolors='none')
        plt.grid(True)
        axs.set_aspect('equal', 'box')
        fig.tight_layout()
        axs.set_title('Constellation of {}'.format(args.file), fontsize=12)
        plt.show()
