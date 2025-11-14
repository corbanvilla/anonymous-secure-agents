#!/bin/bash

echo "Checking connectivity for all VWA_ URLs..."

for var in $(env | grep '^WA_' | grep -v '_TOKEN' | grep -v 'RESET' | cut -d= -f1); do
    url="${!var}"
    echo -n "Curling $var -> $url ... "
    
    # Try to curl with a short timeout
    if curl -L -s --head --request GET "$url" --max-time 5 | grep "200 OK" > /dev/null; then
        echo "Success"
    else
        echo "Failed to connect"
    fi
done
