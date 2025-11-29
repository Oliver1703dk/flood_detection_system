# ~/toggle.sh  (overwrite the old one)
#!/bin/bash
if [[ "$1" == "local" ]]; then
    echo "→ Local cluster mode (MQTT always works)"
    sudo nmcli con up local-static
elif [[ "$1" == "internet" ]]; then
    echo "→ Internet mode — whole cluster gets internet via university wall"
    sudo nmcli con up eth-internet
else
    echo "Usage: $0 [local|internet]"
    echo "Active connection:"
    nmcli con show --active | grep ethernet
fi