#!/usr/bin/env bash

docker build docker/ -f docker/Dockerfile --rm -t $USER/sartorius-cell-instance-segmentation
