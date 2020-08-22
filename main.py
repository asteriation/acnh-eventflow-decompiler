import argparse
import json
from pathlib import Path

from cfg import CFG
import populate_cfg

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Converts .bfevfl files to a readable form')
    parser.add_argument('bfevfl_files', nargs='+', help='.bfevfl file(s) to convert')
    parser.add_argument('--actors', default='actors.json', help='actors.json file for actors\' actions and queries')
    parser.add_argument('--dump-actors', help='dump actors to text files in the specified directory')
    parser.add_argument('--out-dir', help='output directory for .evfl.txt files (default: stdout)')
    args = parser.parse_args()

    with Path(args.actors).open('rt') as af:
        actor_data = json.load(af)

    actors_dumped = False
    for fname in args.bfevfl_files:
        assert fname.endswith('.bfevfl')
        with Path(fname).open('rb') as f:
            print(f'converting {fname}')
            # populate_cfg.parse_bfevfl(f.read())
            cfg = populate_cfg.read(f.read(), actor_data)
            cfg.restructure()

            if args.out_dir:
                with (Path(args.out_dir) / Path(fname).name.replace('.bfevfl', '.evfl.txt')).open('wt') as of:
                    print(cfg.generate_code(), file=of)
            else:
                print(fname)
                print('--------')
                print(cfg.generate_code())
            if args.dump_actors and not actors_dumped:
                for name, actor in cfg.actors.items():
                    with (Path(args.dump_actors) / f'{name}.txt').open('wt') as af:
                        print(actor, file=af)

