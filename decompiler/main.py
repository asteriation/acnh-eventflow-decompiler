import argparse
from collections import OrderedDict
import csv
import distutils.util
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from .logger import LOG

from . import bfevfl
from .actors import HINTS
from .cfg import CFG
from .codegen_evfl import EVFLCodeGenerator

def load_functions_csv(filename: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        with Path(filename).open('rt') as ff:
            reader = csv.reader(ff)
            headers: Dict[str, int] = {}

            for i, col in enumerate(next(reader)):
                headers[col] = i

            required_headers = (
                'Type', 'Name', 'Parameters', 'Return',
                'ConversionKey', 'Conversion', 'NegatedConversion',
            )

            assert all(h in headers for h in required_headers)

            actions: Dict[str, Any] = {}
            queries: Dict[str, Any] = {}

            for row in reader:
                name = row[headers['Name']]
                type_ = row[headers['Type']]

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
                    info['params'][pname] = ptype if not ptype.startswith('Enum') else 'str' # todo: proper EnumXYZ handling

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

        return actions, queries
    except FileNotFoundError:
        LOG.warning(f'{filename} not found, proceeding without function signatures')
        return {}, {}

def main():
    parser = argparse.ArgumentParser(
        description='Converts .bfevfl files to a readable form',
    )
    parser.add_argument('bfevfl_files', nargs='+', help='.bfevfl file(s) to convert')
    parser.add_argument('--functions', default='functions.csv', help='functions.csv file for all actions and queries')
    parser.add_argument('--hints', help='hints.json file for suggestion text based on string parameter values')
    parser.add_argument('--out-dir', help='output directory for .evfl.txt files (default: stdout)')
    parser.add_argument('--target', default='evfl', choices=('evfl',), help='decompilation target')

    bool_check = lambda x: bool(distutils.util.strtobool(x))
    passes = parser.add_argument_group('Primary Decompiler Passes', description='Decompiler passes to improve output code')
    passes.add_argument('--rremove-redundant-switch', metavar='enabled', type=bool_check, default=False, help='Pass to eliminate unconditional switch statements (default: false)')
    passes.add_argument('--rswitch-to-if', metavar='enabled', type=bool_check, default=True, help='Pass to convert switch statements to if/else (default: true)')
    passes.add_argument('--rcollapse-andor', metavar='enabled', type=bool_check, default=True, help='Pass to convert chains of if/else to and/or (default: true)')
    passes.add_argument('--rcollapse-if', metavar='enabled', type=bool_check, default=True, help='Pass to collapse if/else chains to if/elif/else (default: true)')
    passes.add_argument('--rcollapse-case', metavar='enabled', type=bool_check, default=True, help='Pass to combine switchs into group to deduplicate shared end (default: true)')
    passes.add_argument('--rremove-trailing-return', metavar='enabled', type=bool_check, default=True, help='Remove trailing return statements at end of flows (default: true)')

    mpasses = parser.add_argument_group('Secondary Decompiler Passes', description='These passes are run multiple times, after the primary passes')
    mpasses.add_argument('--rextract-reused-blocks', metavar='enabled', type=bool_check, default=True, help='Extract shared code as a subflow (default: true)')
    mpasses.add_argument('--rextract-single-statement', metavar='enabled', type=bool_check, default=False, help='Extract shared single statements as subflows (does nothing if --rextract-reused-blocks (default: false)')
    mpasses.add_argument('--rremove-redundant-entrypoints', metavar='enabled', type=bool_check, default=True, help='Remove redundant entrypoint statements (default: true)')
    mpasses.add_argument('--rcollapse-subflow-only', metavar='enabled', type=bool_check, default=True, help='Collapse roots that only call a single subflow (default: true)')
    mpasses.add_argument('--rsimplify-ifelse-order', metavar='enabled', type=bool_check, default=True, help='Reorder if/else for readability (default: true)')
    mpasses.add_argument('--rsecondary-max-iter', metavar='iters', type=int, default=10000, help='Max number of iterations for secondary passes')
    args = parser.parse_args()

    actions, queries = load_functions_csv(args.functions)

    if args.hints:
        with Path(args.hints).open('rt') as hf:
            HINTS.update(json.load(hf))

    generator = {
        'evfl': EVFLCodeGenerator
    }[args.target]()

    restructure_flags = dict(vars(args))
    for k in list(restructure_flags.keys()):
        if k.startswith('r'):
            restructure_flags[k[1:]] = restructure_flags[k]
        del restructure_flags[k]
    for fname in args.bfevfl_files:
        assert fname.endswith('.bfevfl')
        with Path(fname).open('rb') as f:
            LOG.info(f'converting {fname}')
            cfg = bfevfl.read(f.read(), actions, queries)
            cfg.restructure(**restructure_flags)

            if args.out_dir:
                with (Path(args.out_dir) / Path(fname).name.replace('.bfevfl', '.evfl.txt')).open('wt') as of:
                    print(cfg.generate_code(generator), file=of)
            else:
                print(fname)
                print('--------')
                print(cfg.generate_code(generator))

if __name__ == '__main__':
    main()

