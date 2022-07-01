#!/bin/bash
docker build -t visium ./Dockerfile --no-cache
docker run -p 8888:8888 -p 4444:4444 visium 