#!/usr/bin/env bash

mvn clean package
dot="$(cd "$(dirname "$0")"; pwd)"
export ELEPHAS_CLASS_PATH="$dot/target/elephas-1.0.0-SNAPSHOT-bin.jar"
