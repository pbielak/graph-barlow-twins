#!/bin/bash

docker run --detach \
	       --name "graph-barlow-twins-$(whoami)" \
	       --volume "${PWD}:/app" \
	       --ipc=host \
	       --publish "8888:8888" \
	       --publish "6006:6006" \
	       graph_barlow_twins:latest /bin/bash -c "trap : TERM INT; sleep infinity & wait"
