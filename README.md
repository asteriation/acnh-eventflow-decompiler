# Decompiler for Nintendo EventFlow flowcharts

Tool to decompile eventflow flowcharts (bfevfl) from *Animal Crossing: New Horizons* into a readable, code-like format.

## Usage

The decompiler may be run through `main.py` using Python 3.7.

```bash
mkdir -p out/1.4.0/actors

python3 main.py --actors actors_1.4.0.json \
                --hints hints.json \
                --dump-actors out/1.4.0/actors
                --out-dir out/1.4.0
                romfs/EventFlow/*.bfevfl
```

This dumps the action/query prototypes into `out/1.4.0/actors` and the decompiled evfls into `out/1.4.0`. 

`actors_1.4.0.json` contains a list of actions/queries with type information, which helps with code structuring. If an action/query is not present, the decompiler will attempt to infer types from usage instead.

`hints.json` should contain a mapping of dialog paths (e.g. `Tutorials/Tutorials_Prologue4_Orientation3:045_04_02`) to their contents - these are used to annotate calls with dialogs.

## License

This software is licensed under the terms of the MIT License.
