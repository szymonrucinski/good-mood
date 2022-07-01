#!/bin/sh
echo 'Conda disown'
nohup conda run -n ml python3 app.py &

echo 'Running Jupyter disown'
conda run -n ml jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token='' --NotebookApp.password=''
