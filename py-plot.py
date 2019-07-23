#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot a binary data file')
    parser.add_argument('file', help='full path to data file')
    parser.add_argument('--type', help='data type. Either \'float\' ot \'short\'. Default is float')
    args = parser.parse_args()

    exists = os.path.isfile(args.file)
    if exists:
        if args.type is None or args.type == 'float':
            print 'Reading in as complex-float'
            x = np.fromfile(args.file, dtype=np.complex64)
            t = np.arange(0.0, x.size)
        elif args.type == 'short':
            print 'Reading in as complex-short'
            x = np.fromfile(args.file, dtype=np.int16)
            x = x[1::2] + 1j*x[2::2]
            t = np.arange(0.0, x.size)
        else:
            print 'Unrecognized data type', args.type
    else:
        print 'Could not load file:', args.file
        exit(1)

    # Generate test data file
    # Fs = 100.0
    # N = 1000
    # t = np.arange(0.0, N)/Fs
    # fc = 2.0
    # n = 0.01 * np.matmul(np.random.randn(N, 2), np.array([[1.0], [1.0j]]))
    # n = n.flatten()
    # x = np.exp(2.0j*np.pi*fc*t) + n
    # x.astype(np.complex64).tofile('test_data.bin')

    ax1 = plt.subplot(3, 1, 1)
    plt.plot(t, x.real)
    plt.plot(t, x.imag)
    plt.grid(True)
    plt.title('Real/Imag')
    plt.ylabel('mV')

    ax2 = plt.subplot(3, 1, 2)
    plt.plot(t, np.abs(x))
    plt.title('Abs()')
    plt.grid(True)
    plt.ylabel('mV')

    ax3 = plt.subplot(3, 1, 3)
    plt.magnitude_spectrum(x, Fs=1.0, Fc=0.0, sides='twosided')
    plt.title('Power')
    plt.grid(True)
    plt.ylabel('dB')

    plt.show()
