#!/bin/bash
# run pub/sub sample app using certificates downloaded in package
printf "\nRunning pub/sub sample application...\n"
python MySub.py -e a11nf0pk1jaec3.iot.us-east-1.amazonaws.com -r root-CA.crt -c My209Pi.cert.pem -k My209Pi.private.key
