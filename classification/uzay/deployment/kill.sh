#!/usr/bin/env bash

# (1) Kill torchserve process
torchserve --stop

# (2) Remove model and logging data
# NOTE: You might want to keep these instead in the future...
# rm -r model_store
# rm -r logs