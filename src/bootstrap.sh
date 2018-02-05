#!/bin/bash
for i in {1..1000}; do
	echo "this.py $i &"
	echo "this.py $i &" >> bootstrap.txt
done