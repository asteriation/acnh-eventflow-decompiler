import argparse
import json
from pathlib import Path

from actors import HINTS
from cfg import CFG
import populate_cfg

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Converts .bfevfl files to a readable form')
    parser.add_argument('bfevfl_files', nargs='+', help='.bfevfl file(s) to convert')
    parser.add_argument('--actors', default='actors.json', help='actors.json file for actors\' actions and queries')
    parser.add_argument('--version', default='0.0.0', help='where applicable, actions/queries prefixed with version will be used instead of unprefixed versions')
    parser.add_argument('--hints', help='hints.json file for suggestion text based on string parameter values')
    parser.add_argument('--dump-actors', help='dump actors to text files in the specified directory')
    parser.add_argument('--out-dir', help='output directory for .evfl.txt files (default: stdout)')
    args = parser.parse_args()

    with Path(args.actors).open('rt') as af:
        actor_data = json.load(af)
        for actor_name, data in actor_data.items():
            version_prefixed_actions = [n for n in data['actions'].keys() if '!' in n]
            for name in version_prefixed_actions:
                if name.startswith(f'{args.version}!'):
                    actual_name = name.split('!')[1]
                    if actual_name in data['actions']:
                        data['actions'][actual_name] = data['actions'][name]
                del data['actions'][name]
            version_prefixed_queries = [n for n in data['queries'].keys() if '!' in n]
            for name in version_prefixed_queries:
                if name.startswith(f'{args.version}!'):
                    actual_name = name.split('!')[1]
                    if actual_name in data['queries']:
                        data['queries'][actual_name] = data['queries'][name]
                del data['queries'][name]

    if args.hints:
        with Path(args.hints).open('rt') as hf:
            HINTS.update(json.load(hf))

    actors_dumped = False
    for fname in args.bfevfl_files:
        assert fname.endswith('.bfevfl')
        with Path(fname).open('rb') as f:
            print(f'converting {fname}')
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

