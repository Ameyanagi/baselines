# Fixtures

Golden fixture files in this directory are generated from `pybaselines` for
behavioral comparison only. They must record the `pybaselines` version and must
not include copied implementation code.

Generate fixtures with:

```sh
python3 scripts/generate_pybaselines_fixtures.py
python3 scripts/generate_pybaselines_2d_fixtures.py
```
