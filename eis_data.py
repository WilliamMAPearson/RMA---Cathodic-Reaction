import numpy as np

class EISData:
    def __init__(self, file_names):
        self.eis_data, self.potential_current_data = self.open_eis_data_files(file_names)
        self.frequency_datasets = self.separate_frequency_sets()
        self.extracted_z_data = self.extract_data_from_eis_dataset()

    def open_eis_data_files(self, file_names): # load EIS data files as np array of arrays
        np_array_list = []
        potential_current_data = None
        for eis_file in file_names:
            loaded_data = np.loadtxt(eis_file, delimiter='\t')
            if loaded_data.shape[1] == 2:
                potential_current_data = loaded_data
            else:
                np_array_list.append(loaded_data)
        eis_data = np.array(np_array_list, dtype='object')
        return eis_data, potential_current_data

    def separate_frequency_sets(self):
        frequency_datasets_list = []
        for eis_dataset in self.eis_data:
            frequency_dataset = eis_dataset[:, 0]
            frequency_datasets_list.append(frequency_dataset)
        return frequency_datasets_list

    def extract_data_from_eis_dataset(self):
        extracted_data_list = []
        for eis_dataset in self.eis_data:
            extracted_data_list.append(np.vectorize(complex)(eis_dataset[:, 1], eis_dataset[:, 2]))
        extracted_data = np.array(extracted_data_list, dtype='object')
        return extracted_data