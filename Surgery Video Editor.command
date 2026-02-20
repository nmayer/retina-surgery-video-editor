#!/bin/bash
cd "$(dirname "$0")"
sleep 1 && open http://localhost:5555 &
python3 app.py
