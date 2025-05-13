# !/usr/env/bash

kill `pgrep demo`

if [[ $# -eq 0 ]];then 
  exit 0
fi

msg_len=$2
echo msg_len=${msg_len:=65536}

UCX_TLS=rc_x UCX_NET_DEVICES=mlx5_bond_0:1

if [[ $1 -lt 0 ]];then
  echo disable log..
  taskset -c 31,33 test/apps/iodemo/io_demo -t1200 -n1200 -d$msg_len > /dev/null &
elif [[ $# -lt 3 ]];then
  echo log into stdout
  taskset -c 31,33 test/apps/iodemo/io_demo -t1200 -n1200 -d$msg_len >&1 &
else
  echo log into $3
  taskset -c 31,33 test/apps/iodemo/io_demo -t1200 -n1200 -d$msg_len > $3 &
fi
#UCX_TLS=dc_x UCX_NET_DEVICES=mlx5_bond_0:1 numactl -i netdev:ens7f0np0 test/apps/iodemo/io_demo -d65536 &
