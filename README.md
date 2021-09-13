# Decompiler for Nintendo EventFlow flowcharts

Tool to decompile eventflow flowcharts (bfevfl) from *Animal Crossing: New Horizons* into a readable, code-like format.

## Usage

The decompiler may be run through `main.py` using Python 3.7.

You will need to supply a `functions.csv` file containing typing information for EventFlow functions; this can be done for ACNH by downloading the 'Actions/Queries' sheet from [this
spreadsheet](https://docs.google.com/spreadsheets/d/1AYM-UeRkbJuGy_nKv7AMngevwBtMdZPtfoHEQev8BhM/edit) as a CSV file. This file is used to improve decompilation results.

You may also optionally supply a `hints.json` file, which should contain a string to string mapping. Whenever a key in the mapping is used as a parameter to a EventFlow action or query, a comment will be generated above with the corresponding value. This is useful for annotating dialogue string references with the corresponding dialogue, for example.

And finally, you will also need to supply the bfevfl files to be decompiled.

```bash
mkdir -p out/

python3 main.py --functions functions.csv \
                --version 1.0.0 \
                --hints hints.json \
                --out-dir out/1.0.0 \
                romfs/EventFlow/*.bfevfl
```

This outputs the decompiled evfls into `out/1.0.0`.

There are also a handful of flags starting with `--r` (use `--help` for a full list) that control which decompiler passes are run.

## License

This software is licensed under the terms of the MIT License.
