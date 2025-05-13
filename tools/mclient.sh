echo '[connect num][round][-d][-i][addr]'
epn=$1
round=$2
msg_length=$3
iter=$4
addr=$5

if [[ $# -ge 6 ]]; then
  echo log into $6
fi

epn=${epn:=1}
round=${round:=1}
msg_length=${msg_length:=4096}
iter=${iter:=0}
addr=${addr:='192.168.11.13'}

servers=""
for i in `seq 1 $epn`
do
servers="${servers} ${addr}"
done

#bin_path="test/apps/iodemo"
bin_path="install/bin"

UCX_TLS=dc_x UCX_NET_DEVICES=mlx5_bond_0:1 
UCX_RC_ROCE_PATH_FACTOR=2
UCX_DC_ROCE_PATH_FACTOR=2

for i in `seq 1 $round`;do

# $bin_path/io_demo -d $msg_length -i $iter $servers
# numactl -i netdev:ens7f0np0 $bin_path/io_demo -d $msg_length -i $iter $servers
# taskset -c 11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41 $bin_path/io_demo -d $msg_length -i $iter $servers
# taskset -c 32,34 $bin_path/io_demo -d $msg_length -i $iter $servers
if [[ $# -lt 6 ]]; then
taskset -c 31,33 $bin_path/io_demo -d $msg_length -i $iter $servers -n1200 -t1200
else
UCX_TLS=dc_x UCX_NET_DEVICES=mlx5_bond_0:1 taskset -c 31,33 $bin_path/io_demo -d $msg_length -i $iter $servers -n1200 -t1200 > $6
fi

done
