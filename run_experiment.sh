#!/bin/bash

# dataset: aquatic
uv run main.py model=wildlife dataset=aquatic
uv run main.py model=miewid dataset=aquatic
uv run main.py model=aliked dataset=aquatic
uv run main.py model=disk dataset=aquatic
uv run main.py model=sift dataset=aquatic

# dataset: redang
uv run main.py model=wildlife dataset=redang
uv run main.py model=miewid dataset=redang
uv run main.py model=aliked dataset=redang
uv run main.py model=disk dataset=redang
uv run main.py model=sift dataset=redang


# dataset: SeaTurtleIDHeads
uv run main.py model=wildlife dataset=seaturtleidheads
uv run main.py model=miewid dataset=seaturtleidheads
uv run main.py model=aliked dataset=seaturtleidheads
uv run main.py model=disk dataset=seaturtleidheads
uv run main.py model=sift dataset=seaturtleidheads
