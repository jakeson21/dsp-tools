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
            print 'Got float'
            x = np.fromfile(args.file, dtype=np.complex64)
        elif args.type == 'short':
            print 'Got short'
        else:
            print 'Unrecognized data type', args.type
    else:
        print 'Could not load file:', args.file
        exit(1)

    ax1 = plt.subplot(3, 1, 1)
    plt.title('Real/Imag')
    plt.ylabel('mV')

    ax2 = plt.subplot(3, 1, 2)
    plt.title('Abs()')
    plt.ylabel('mV')

    ax3 = plt.subplot(3, 1, 3)
    plt.title('Power')
    plt.ylabel('dB')

    plt.show()
