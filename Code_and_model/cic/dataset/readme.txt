Preprocessing by the follow tasks:

1. concat all days dataset
2. change port number by :
    port 0-1023 -> 1
    prot 1024-49151 -> 2
    port 49152 - 65535 -> 3
3. OneHotEncoder the port number column