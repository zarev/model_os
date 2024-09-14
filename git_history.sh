#!/bin/bash

git log -n 10 --pretty=format:'%h - %an, %ad : %s' --date=short
