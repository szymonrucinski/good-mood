docker build -t visium --platform linux/amd64 . --no-cache
docker run -p 8888:8888 visium 