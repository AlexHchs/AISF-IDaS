import pandas as pd
import sys
import os
from dateutil.parser import parse


def main(argv):
    if len(argv) != 4:
        print("Usage: {} raw_input_file output_csv_file delete_filter_file(0/1, default=1)".format(argv[0]))
        sys.exit()

    raw_input_file = argv[1]
    output_csv_file = argv[2]
    filter_file = output_csv_file[:-4] + '_filter.txt'
    delete_filter_flag = int(argv[3])
    if delete_filter_flag != 0 and delete_filter_flag != 1:
        delete_filter_flag = 1

    with open(raw_input_file, 'r') as fp:
        with open(filter_file, 'a') as wfp:
            cmd_position = None
            timestamp = None

            # head
            line = fp.readline()
            while 'PID' not in line:
                if 'ATOP' in line:
                    temp = line.split()
                    """
                    fix format error in head one, should be 8 items here

                    error format ex: 
                    ATOP - benign-server2022/11/01  09:34:56------     5d17h31m55s elapsed
                    
                    correct format ex:
                    ATOP - benign-server     2022/11/01  09:34:56     ------     5d17h31m55s elapsed
                    """
                    if len(temp) == 6:
                        tmp2 = temp[2][:-10]
                        tmp3 = temp[2][-10:]
                        tmp4 = temp[3][:8]
                        tmp5 = temp[3][8:]
                        tmp6 = temp[4]
                        tmp7 = temp[5]
                        temp[2] = tmp2
                        temp[3] = tmp3
                        temp[4] = tmp4
                        temp[5] = tmp5
                        temp.append(tmp6)
                        temp.append(tmp7)

                    time_str = temp[3] + " " + temp[4]
                    dt = parse(time_str)
                    timestamp = dt.timestamp()

                line = fp.readline()
            
            # print(line)
            cmd_position = line.find("CMD")
            wfp.write("TIMESTAMP " + line)
            
            # body
            wflag = 1
            line = fp.readline()
            while line:
                if 'ATOP' in line or line == '\n':
                    wflag = 0
                    if 'ATOP' in line:
                        temp = line.split()
                        time_str = temp[3] + " " + temp[4]
                        dt = parse(time_str)
                        timestamp = dt.timestamp()

                elif 'PID' in line:
                    wflag = 1
                    line = fp.readline()
                
                if wflag == 1:
                    cmd_string = line[cmd_position:].strip()
                    cmd_string_replace = cmd_string.replace(' ', '_')
                    line = line.replace(cmd_string, cmd_string_replace)
                    wfp.write("{} ".format(timestamp) + line)

                line = fp.readline()

    df = pd.read_csv(filter_file, delim_whitespace=True)
    df.to_csv(output_csv_file, index=False)
    
    if delete_filter_flag:
        os.remove(filter_file)


if __name__ == '__main__':
    main(sys.argv)
