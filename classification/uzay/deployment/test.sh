#!/usr/bin/env bash

# (1) Send prediction and get request
curl -X POST http://127.0.0.1:8080/predictions/oraquick_classifier -T IMG_20190614_161501.jpg
