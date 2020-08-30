import argparse
from collections import OrderedDict
import csv
import json
from pathlib import Path
from typing import Any, Dict

from logger import LOG

from actors import HINTS
from cfg import CFG
import populate_cfg

def compare_version(current_version: str, max_version: str) -> bool:
    cv = [int(v) for v in current_version.split('.')]
    mv = [int(v) for v in max_version.split('.')]
    assert len(cv) == len(mv)

    for c, m in zip(cv, mv):
        if c < m:
            return True
        if c > m:
            return False

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Converts .bfevfl files to a readable form')
    parser.add_argument('bfevfl_files', nargs='+', help='.bfevfl file(s) to convert')
    parser.add_argument('--functions', default='functions.csv', help='functions.csv file for all actions and queries')
    parser.add_argument('--version', default='0.0.0', help='where applicable, actions/queries prefixed with version will be used instead of unprefixed versions')
    parser.add_argument('--hints', help='hints.json file for suggestion text based on string parameter values')
    parser.add_argument('--out-dir', help='output directory for .evfl.txt files (default: stdout)')
    args = parser.parse_args()

    with Path(args.functions).open('rt') as ff:
        reader = csv.reader(ff)
        headers: Dict[str, int] = {}

        for i, col in enumerate(next(reader)):
            headers[col] = i

        required_headers = (
            'MaxVersion', 'Type', 'Name', 'Parameters', 'Return',
            'ConversionKey', 'Conversion', 'NegatedConversion',
        )

        assert all(h in headers for h in required_headers)

        actions: Dict[str, Any] = {}
        queries: Dict[str, Any] = {}

        for row in reader:
            max_version = row[headers['MaxVersion']]
            name = row[headers['Name']]
            type_ = row[headers['Type']]

            if max_version == 'pseudo' or not compare_version(args.version, max_version):
                continue
            if (type_ == 'Action' and name in actions) or (type_ == 'Query' and name in queries):
                continue

            info: Dict[str, Any] = {}

            param_info = row[headers['Parameters']].split(';') if row[headers['Parameters']] else []
            info['params'] = OrderedDict()
            for param in param_info:
                param = param.strip()
                assert ':' in param and param.index(':') not in (0, len(param) - 1), f'bad param list for {name}'
                pname, ptype = (x.strip() for x in param.split(':'))
                assert ptype != 'inverted_bool', f'inverted_bool not allowed for param types for {name}'
                info['params'][pname] = ptype

            if row[headers['Return']]:
                info['return'] = row[headers['Return']]
                if info['return'] == 'inverted_bool':
                    info['return'] = 'bool'
                    info['inverted'] = True

            if row[headers['ConversionKey']]:
                if row[headers['Conversion']]:
                    info['conversion'] = {
                        'key': row[headers['ConversionKey']],
                        'values': [x.strip() for x in row[headers['Conversion']].split('\n')]
                    }
                if row[headers['NegatedConversion']]:
                    info['neg_conversion'] = {
                        'key': row[headers['ConversionKey']],
                        'values': [x.strip() for x in row[headers['NegatedConversion']].split('\n')]
                    }
            else:
                if row[headers['Conversion']]:
                    info['conversion'] = row[headers['Conversion']]
                if row[headers['NegatedConversion']]:
                    info['neg_conversion'] = row[headers['NegatedConversion']]

            if type_ == 'Action':
                actions[name] = info
            elif type_ == 'Query':
                queries[name] = info
            else:
                raise ValueError(f'bad function type: {type_}')

    if args.hints:
        with Path(args.hints).open('rt') as hf:
            HINTS.update(json.load(hf))

    for fname in args.bfevfl_files:
        assert fname.endswith('.bfevfl')
        with Path(fname).open('rb') as f:
            LOG.info(f'converting {fname}')
            cfg = populate_cfg.read(f.read(), actions, queries)
            cfg.restructure()

            if args.out_dir:
                with (Path(args.out_dir) / Path(fname).name.replace('.bfevfl', '.evfl.txt')).open('wt') as of:
                    print(cfg.generate_code(), file=of)
            else:
                print(fname)
                print('--------')
                print(cfg.generate_code())

