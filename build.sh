#!/bin/bash
mkdir -p $HOME/.pdl4j
cp pom.xml $HOME/.pdl4j
sudo docker build -f JavaDockerfile . -t pydl4j
sudo docker run --mount src="$HOME/.pydl4j",target=/app,type=bind pydl4j
export ELEPHAS_CLASS_PATH=$HOME./pydl4j