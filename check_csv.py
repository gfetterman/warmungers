import argparse
import bark
import sys

def _parse_args(raw_args):
    desc = 'Sanity-check Bark event dataset against a sampled dataset.'
    p = argparse.ArgumentParser(description=desc)
    p.add_argument('event_dset', help='Bark event dataset to sanity-check')
    p.add_argument('out', help='name of output event dataset')
    p.add_argument('sampled', help='Bark sampled dataset')
    p.add_argument('-b',
                   '--beginning',
                   help='action on labels before 0 (default: truncate)',
                   choices=['ignore', 'truncate', 'delete'],
                   default='truncate')
    p.add_argument('-r',
                   '--reverse',
                   help='action on labels with stop < start (default: flip)',
                   choices=['ignore', 'flip', 'delete'],
                   default='flip')
    p.add_argument('-e',
                   '--end',
                   help='action on labels after end (default: truncate)',
                   choices=['ignore', 'truncate', 'delete'],
                   default='truncate')
    return p.parse_args(raw_args)

def check_csv(event_dset, sampled, out, beginning, reverse, end):
    events = bark.read_events(event_dset)
    df = events.data
    if beginning == 'truncate':
        df['start'] = df['start'].apply(lambda x: max(0, x))
        df['stop'] = df['stop'].apply(lambda x: max(0, x))
    elif beginning == 'delete':
        df = df[df['start'] >= 0]
        df = df[df['stop'] >= 0]
    else: # beginning == 'ignore'
        pass
    if reverse == 'flip':
        def flip(row):
            if row['stop'] < row['start']:
                row['stop'], row['start'] = row['start'], row['stop']
            return row
        df = df.apply(flip, axis='columns')
    elif reverse == 'delete':
        df = df[df['stop'] >= df['start']]
    else: # reverse == 'ignore'
        pass
    if end != 'ignore': # only open the sampled dataset if necessary
        sampled_dset = bark.read_sampled(sampled)
        dset_end = sampled_dset.data.shape[1]
        if events.attrs['columns']['start']['units'] == 's':
            dset_end = dset_end / sampled_dset.attrs['sampling_rate']
        if end == 'truncate':
            df['start'] = df['start'].apply(lambda x: min(dset_end, x))
            df['stop'] = df['stop'].apply(lambda x: min(dset_end, x))
        elif end == 'delete':
            df = df[df['start'] <= dset_end]
            df = df[df['stop'] <= dset_end]
    return bark.write_events(out, df, **events.attrs)

if __name__ == '__main__':
    p = _parse_args(sys.argv[1:])
    _ = check_csv(p.event_dset, p.sampled, p.out, p.beginning, p.reverse, p.end)

