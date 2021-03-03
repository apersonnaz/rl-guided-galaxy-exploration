#!/bin/bash

tar -czh . | sudo docker build -t myimage -
