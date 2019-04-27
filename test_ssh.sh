#!/bin/bash

touch test.txt
git add test.txt
git commit -m "Testing ssh from script"
git push

echo Done!

