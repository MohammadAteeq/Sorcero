class FileReader:
    def read_raw_file(self, file_name):
        fp = open(file_name)
        data = fp.read()
        return data
