#!/bin/bash

GPU="${1}"

docker run --detach \
	       --name "graph-barlow-twins-$(whoami)" \
	       --volume "${HOME}/graph-barlow-twins:/app" \
	       --gpus "device=${GPU}" \
	       --ipc=host \
	       --publish "30982:8888" \
	       --publish "30983:6006" \
	       graph_barlow_twins:latest /bin/bash -c "trap : TERM INT; sleep infinity & wait"
