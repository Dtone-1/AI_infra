#/bin/bash
export MY_CONTAINER="$(whoami)_dev_wrksp"
num=`docker ps -a|grep -w "$MY_CONTAINER$"|wc -l`
echo $num
if [ 0 -eq $num ];
then
  echo "new docker"
  docker run -it --name=$MY_CONTAINER --network=host \
    --shm-size 20g \
    --cap-add=sys_ptrace \
    -v /workspace:/workspace/volume \
    -v /tmp:/workspace/tmp \
    -v /usr/bin/cnmon:/usr/bin/cnmon \
    -v /ittest:/ittest \
    -v /data:/data -v /edge_test:/edge_test -w /workspace \
    -v /data2:/data2 \
    -v /data1:/data1 \
    -v /projs:/projs \
    --privileged \
    --device=/dev/cambricon_ctl \
    --device=/dev/dri \
    yellow.hub.cambricon.com/cambricon_pytorch_container/cambricon_pytorch_container:v26.06.0-torch2.12.0-torchmlu1.33.1-ubuntu22.04-py310  \
    /bin/bash
else
  echo "docker start"
  docker start $MY_CONTAINER
  docker exec -ti $MY_CONTAINER /bin/bash
fi
