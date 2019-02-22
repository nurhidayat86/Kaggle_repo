import h5py
import pandas as pd

class feature_h5:

    def __init__(self, h5_path):
        self.h5_path = h5_path

    def feature_cwtf_load(self):
        file = h5py.File(self.h5_path, 'r')
        feature = file['feature_cwtf'][:,:]
        column_x = [f"cwt{i}" for i in range(0, feature.shape[1])]
        file.close()
        feature = pd.DataFrame(columns=column_x, data=feature)
        return feature

    def feature_f_load(self):
        file = h5py.File(self.h5_path, 'r')
        feature = file['feature_f'][:,:]
        column_x = [f"f{i}" for i in range(0, feature.shape[1])]
        file.close()
        feature = pd.DataFrame(columns=column_x, data=feature)
        return feature

    def feature_t_load(self):
        file = h5py.File(self.h5_path, 'r')
        feature = file['feature_t'][:,:]
        file.close()
        column_x = [f"t{i}" for i in range(0, feature.shape[1])]
        feature = pd.DataFrame(columns=column_x, data=feature)
        return feature

    def feature_desc_load(self):
        file = h5py.File(self.h5_path, 'r')
        feature = file['feature_desc'][:,:]
        column_x = ['mean', 'std', 'var', 'max', 'sum']
        feature = pd.DataFrame(columns=column_x, data=feature)
        file.close()
        return feature

    def y_load(self):
        file = h5py.File(self.h5_path, 'r')
        ttf = file['ttf'][:,:]
        column_x = ['ttf']
        file.close()
        ttf = pd.DataFrame(columns=column_x, data=ttf)
        return ttf


if __name__ == "__main__":
    pass
    # feat = feature_h5("H:\\kaggle\\LANL-Earthquake-Prediction\\train.h5")
    # feature = feat.feature_load()
    # y = feat.y_load()
    #
    # print(f"shape feature: {feature.shape}, y: {y.shape}")