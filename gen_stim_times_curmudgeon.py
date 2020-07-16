import argparse
import bark
import numpy as np
import pandas as pd
from scipy.io import wavfile
import sys
import os.path

THRESHOLD = 20000.


def _parse_args(raw_args):
    desc = 'Combine jill triggers and jstim output to make stimulus time .csv'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-o', '--out', help='output file')
    parser.add_argument('pulse', help='bark dataset with pulse channel')
    parser.add_argument('log', help='logfile')
    parser.add_argument('-w',
                        '--wav',
                        nargs='+',
                        help='WAV files containing the stimuli')
    parser.add_argument('-c',
                        '--channel',
                        help='channel containing pulse, default: 0',
                        type=int,
                        default=0)
    return parser.parse_args(raw_args)


def gen_stim_times(pulse_file, log_file, out_file, wav_files, channel):
    pulse = bark.read_dataset(pulse_file)

    # pull starts from ADC trigger channel and convert to seconds
    above_thresh = (pulse.data[:, channel] >= THRESHOLD).astype(int)
    starts = (np.nonzero(np.diff(above_thresh) > 0)[0])
    starts = starts / pulse.attrs['sampling_rate']

    # pull stimulus order and names from jstim log
    logfile = open(log_file, 'r').readlines()
    stims = [s.split(' ')[4] for s in logfile if 'next stim: ' in s]

    # determine stimulus file lengths
    lengths = {}
    for wav_file in wav_files:
        sr, data = wavfile.read(wav_file)
        stim_name = os.path.splitext(os.path.basename(wav_file))[0]
        lengths[stim_name] = len(data) / sr

    stops = [start + lengths[name] for start, name in zip(starts, stims)]

    # write stimulus intervals to Bark event dataset
    stimdata = pd.DataFrame({'start': starts, 'name': stims, 'stop': stops})
    attrs = {'columns': {'name': {'units': None},
                         'start': {'units': 's'},
                         'stop': {'units': 's'}},
             'datatype': 2001}

    bark.write_events(out_file, stimdata, **attrs)


def _main():
    args = _parse_args(sys.argv[1:])
    gen_stim_times(args.pulse, args.log, args.out, args.wav, args.channel)


if __name__ == '__main__':
    _main()
