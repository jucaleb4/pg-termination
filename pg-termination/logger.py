import numpy as np

class BasicLogger():
    def __init__(self, fname, keys, dtypes):
        """
        :params fname: filename we want to save 
        :params keys: keys when printing
        :param dtypes: dtypes of the file
        """
        self.fname = fname
        self.keys = keys
        self.dtypes = dtypes
        n = 1024
        self.data = np.zeros((n, len(self.keys)), dtype=float)
        self.ct = 0

    def log(self, *args):
        assert len(args) == len(self.keys), "Number of inputs %d does not match number of keys %d" % (len(args), len(self.keys))

        self.data[self.ct] = args

        self.ct += 1
        if self.ct == len(self.data):
            self.data = np.vstack((self.data, np.zeros(self.data.shape, dtype=self.data.dtype)))

    def save(self):
        if not self.fname.endswith(".csv"):
            self.fname += ".csv"
        fp = open(self.fname, "w+")
        fmt = ','.join(["%d" if self.dtypes[i] == "d" else "%.6e" for i in range(len(self.dtypes))])
        fp.write("%s\n" % (','.join(self.keys)))
        m = self.data.shape[1]
        for i in range(self.ct):
            for j in range(m):
                fp.write(("%d" if self.dtypes[j] == "d" else "%.6e") % self.data[i,j])
                if j < m-1:
                    fp.write(",")
                else:
                    fp.write("\n")
        fp.close()    
