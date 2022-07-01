#!/bin/bash
docker build -t visium --platform linux/amd64 ./Dockerfile --no-cache
docker run -p 8888:8888 -p 4444:4444 visium 