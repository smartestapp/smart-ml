#!/usr/bin/env bash

# (1) Send prediction and get request
counter=1
while [ $counter -le 77 ]
do
	echo $counter
	curl -X POST http://127.0.0.1:8080/predictions/btnx_outlier -T test_images/{$counter}.jpg
	((counter++))
done
