#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from matplotlib.animation import FuncAnimation
from scipy import signal
from scipy import interpolate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot a binary file with complex short or float data')
    parser.add_argument('file', help='full path to data file')
    parser.add_argument('fs', help='Sample rate of data in Hz')
    parser.add_argument('-t', '--type', help='data type. Either \'float\' ot \'short\'. Default is float')
    parser.add_argument('--sps', help='the number of samples per symbol. Default is 2')
    parser.add_argument('-a', '--axes-limit', help='Sets ± axes limits. Default is 1.0')
    parser.add_argument('-d', '--downsample', help='The number of Symbols to downsample by. Default is 1')
    parser.add_argument('-iq', action='store_true', default=False, help='Make IQ constellation plot. Default is Power vs. Time.')
    parser.add_argument('-s', '--start', help='Sample index to start plotting with. Default is 0')
    parser.add_argument('-q', action='store_true', default=False, help='Exit when data is exhausted')
    args = parser.parse_args()
    print(np.complex64.nbytes)
    fs = int(args.fs)
    print('fs = {}'.format(fs))

    f_type = np.complex64
    exists = os.path.isfile(args.file)
    if exists:
        if args.type is None or args.type == 'float':
            print('Reading in as complex-float')
            f_type = np.complex64
        elif args.type == 'short':
            print('Reading in as complex-short')
            f_type = np.int16
        else:
            print('Unrecognized data type', args.type)
    else:
        print('Could not load file: {}'.format(args.file))
        exit(1)

    sps = 2
    if args.sps is not None and int(args.sps)>0:
        sps = int(args.sps)
    print('Sps = {}'.format(sps))
    
    scale = 1.0
    if args.axes_limit is not None and float(args.axes_limit)>0:
        scale = float(args.axes_limit)
    print('Scale = {}'.format(scale))

    skip = 1
    if args.downsample is not None and int(args.downsample)>0:
        skip = int(args.downsample)
    print('Downsample = {}'.format(skip))
    
    start = 0
    if args.start is not None and int(args.start)>0:
        start = int(args.start)
    print('Start index = {}'.format(start))
    
    quit = args.q

    file_name = args.file
    fid = open(file_name, "rb")
    print('Opened {}'.format(file_name))

    x, y = np.ndarray((0,)), np.ndarray((0,))
    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(1, 1, figsize=(3,3))
    if args.iq:
        # sc = ax.scatter(x, y, alpha=0.25, edgecolors='none')
        sc = ax.plot(x, y, '.')
        ax.set_aspect('equal', 'box')
        ax.set_title('Constellation of\n{}'.format(file_name), fontsize=10)
        ax.grid(True)
        plt.xlim(-scale, scale)
        plt.ylim(-scale, scale)
    else:
        sc = ax.plot(x, y)
        ax.set_title('Power vs. Time of\n{}'.format(file_name), fontsize=10)
        ax.grid(True)
        plt.xlim(0., 1.)
        plt.ylim(-scale, scale)

    def init():
        if args.iq:
            #sc.set_offsets(np.c_[x, y])
            sc[0].set_xdata(x)
            sc[0].set_ydata(y)
            return sc
        else:
            sc[0].set_xdata(x)
            sc[0].set_ydata(y)

    def animate(i):
        global x, y, t
        N = int(fs/10)
        z = np.fromfile(fid, count=N, dtype=f_type)
        print(i, N, z.size, fid.tell(), sps)
        if z.size > 0:
            if z.size != N:
                # Rewind and wait for a full N samples
                fid.seek(1, -z.nbytes)
                print('not enough samples...')
            elif args.iq:
                x = z.real[start::sps*skip]
                y = z.imag[start::sps*skip]
                # sc.set_offsets(np.c_[x, y])
                sc[0].set_xdata(x)
                sc[0].set_ydata(y)
            else:
                z_dB = 20.*np.log10(np.abs(z[::skip]))
                if x.size != z_dB.size:
                    x = np.arange(z_dB.size)
                    plt.xlim(x[0], x[-1])
                    sc[0].set_xdata(x[start::skip])
                sc[0].set_ydata(z_dB)
        else:
            if not args.q:
                print('waiting for samples...')
            else:
                plt.close(fig)
        return z
                
    ani = FuncAnimation(fig, animate, interval=50, repeat=False)
    plt.show()
    fid.close()
