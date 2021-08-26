from a2_chooseTestData import *




if __name__ == "__main__":
    file = open("data/20200417/test_100K12_multi.txt")
    test_dir = "data/20200417"
    save_dir = "vwccTestDate"
    filenums = [0, 0, 0, 0, 0, 0]
    n = 0
    for line in file:
        line = line.split()
        if int(line[1]) == 0:
            copyFile(test_dir + line[0], save_dir + "/dianhua0" + line[0])
            filenums[0] += 1
        elif int(line[1]) == 1:
            copyFile(test_dir + line[0], save_dir + "/dianhua1" + line[0])
            filenums[1] += 1
        elif int(line[1]) == 2:
            copyFile(test_dir + line[0], save_dir + "/dianhua2" + line[0])
            filenums[2] += 1
        elif int(line[1]) == 3:
            copyFile(test_dir + line[0], save_dir + "/dianhua3" + line[0])
            filenums[3] += 1

        if int(line[2]) == 0:
            copyFile(test_dir + line[0], save_dir + "/chouyan0" + line[0])
            filenums[4] += 1
        elif int(line[2]) == 1:
            copyFile(test_dir + line[0], save_dir + "/chouyan1" + line[0])
            filenums[5] += 1
        n += 1
        if n % 10000 == 0:
            print("Have processed %d images." % (n))
    print(filenums)
