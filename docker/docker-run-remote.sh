docker run -p 9001:9001 \
    -w /dim-bridge \
    -v "$(pwd)/datasets:/dim-bridge/datasets" \
    tiga1231/dim-bridge
