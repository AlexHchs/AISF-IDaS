if [ $# != 3 ]; then
    echo "Usage: ./make_subflow_dataset.sh time_window(secs) pcap_file_path code_path"
    exit 1
fi

time_window=$1
pcap_file_path=$2
code_path=$3

cd $pcap_file_path
mkdir -p "pcap_${time_window}secs"

for pcapfile in $(ls *.pcap); do
    # # split to subflow
    # tshark -r $filename -Y "frame.time_relative >= ${lowerbound} && frame.time_relative <= ${upperbound}" -w "pcap_${time_window}secs/${newName}.pcap"
    
    filename=$(basename $pcapfile .pcap)

    # extract feature
    argus -r $pcapfile -w "pcap_${time_window}secs/${filename}.argus" -S $time_window
    ra -unzr "pcap_${time_window}secs/${filename}.argus" -c , -s rank stime flgs proto saddr sport daddr dport pkts bytes state ltime seq dur mean stddev sum min max spkts dpkts sbytes dbytes rate srate drate > "pcap_${time_window}secs/${filename}.csv"
    
    # delete subflow argus file to save space
    rm "pcap_${time_window}secs/${filename}.argus"
done

cd "pcap_${time_window}secs"
# merge subflow
python3 "${code_path}/merge_subflow_csv.py" "${time_window}secs"
cd ..