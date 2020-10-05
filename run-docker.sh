docker run --rm -it --init --gpus=all --ipc=host --volume="$PWD:/usr/local/app" taher/pytorch bash

# Use crl+p+q to detach without killing
# Use docker ps to see the ids
# use docker attach id 