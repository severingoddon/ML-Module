with open("diagnosis.data",'r', encoding="utf16") as fFloat:
    with open("diagnosis.csv","w") as fString:
        for line in fFloat:
            line = line.replace(',', '.')
            line = line.replace('\t', ',')
            line = line.replace('yes', '1')
            line = line.replace('no', '0')
            line = line.replace('\r\n', '\n')
            fString.write(line)

