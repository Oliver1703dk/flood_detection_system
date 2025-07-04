#!/bin/bash

if [[ "$1" == "internet" ]]; then
    echo "Switching to Ethernet Internet Mode..."
    sudo nmcli con down eth-static
    sudo nmcli con up eth-internet
elif [[ "$1" == "static" ]]; then
    echo "Switching to Ethernet Static Pi-to-Pi Mode..."
    sudo nmcli con down eth-internet
    sudo nmcli con up eth-static
else
    echo "Usage: ./toggle-eth.sh [internet|static]"
fi

