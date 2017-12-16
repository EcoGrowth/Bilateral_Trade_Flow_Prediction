import numpy as np




# Takes the log of features
def convertToLog(x, y, log_transform_list = ['GDP_o', 'GDP_d', 'POP_o',
                                             'POP_d', 'Dist_coord', 'XPTOT_o']):
    y = y.apply(np.log)
    for col in log_transform_list:
        x[col] = x[col].apply(np.log)

    return x, y