# terracanopy
Software to analyze images of forest growth to track forest health

Band-aware, streaming analysis: phone photos, satellite tiles, and (later) real-time
multispectral drone video all become the same `Frame` (bands, H, W) and flow through
`source -> analyzers -> sinks`.

## Getting started (Python 3.15, managed with [uv](https://docs.astral.sh/uv/))

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh   # once per machine
uv sync                                           # creates .venv, installs exact locked versions + dev tools
uv run pre-commit install                         # once per clone: lint/format on commit
uv run pytest                                     # run tests
```

`uv sync` reads `pyproject.toml` and `uv.lock`, installs Python 3.15 if needed (see
`.python-version`), and installs the project in editable mode. Optional features:
`uv sync --extra geo`.

Without uv, plain pip works too (pip >= 25.1):

```sh
python3.15 -m venv .venv && source .venv/bin/activate
pip install -e . --group dev
```

## Everyday commands

```sh
uv run pytest                 # tests
uv run ruff check --fix .     # lint
uv run ruff format .          # format
uv run pyright                # type check
uv add <package>              # add a dependency (updates pyproject.toml and uv.lock)
uv add --group dev <package>  # add a dev-only tool
uv lock --upgrade             # refresh pinned versions
```

Commit `uv.lock`. It is what makes everyone's environment identical.

If installs try to compile numpy or opencv from source, binary wheels for this Python
version may not be published yet. Check with
`uv pip install --only-binary=:all: numpy opencv-python-headless`.

## Usage

```sh
terracanopy indices                       # list spectral indices and the bands they need
terracanopy analyze photo.jpg             # one photo
terracanopy analyze photos/ --out out     # folder of photos
terracanopy analyze flight.mp4 --stride 5 # video, every 5th frame
terracanopy analyze rtsp://host/stream    # live stream
```

Output: `out/results.jsonl` (per-frame metrics incl. latency) and `out/overlays/*.png`.

## Layout

- `frame.py`: `Frame`, the single data type (canonical band names)
- `indices.py`: ExG, VARI, NDVI, NDRE, GNDVI, NDWI; each declares required bands
- `analyzers/`: plugin interface plus the baseline `VegetationAnalyzer`
- `sources.py`: images, directories, video files, camera/RTSP streams
- `sinks.py`: JSONL metrics and PNG overlays
- `pipeline.py`: wires them together and records per-frame latency
