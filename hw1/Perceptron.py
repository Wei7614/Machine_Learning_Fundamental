import numpy as np


class Perceptron:

    def __init__(self):
        pass

    # 定義activate function
    def __sign(self, z):
        if z > 0:
            return 1
        else:
            return -1

    # 實作perceptron learning algorithm
    def get_weight(self, data, y_set, eta=1, total_run_time=1, is_random_seed=False, update_limit=-1):
        # w加1是因為需要有w_0
        w = np.zeros(data.shape[1] + 1)
        for run in range(total_run_time):
            print("run " + str(run))
            if is_random_seed:
                w = np.random.rand(1, data.shape[1] + 1)
            update = 0
            out_off_loop = False
            while not out_off_loop:
                #每一筆資料去比對
                out_off_loop = True
                for idx in range(data.shape[0]):
                    x = np.concatenate((np.array([1.]), data[idx, :]))
                    y = y_set[idx]
                    if self.__sign(np.dot(w, x)) != y:
                        # 遇到錯的就更新並且將w做修正，修正過後的w還要重新train一次所以out_off_loop要設為false
                        out_off_loop = False
                        update += 1
                        w = w + eta * y * x
                if update_limit != -1 and update >= update_limit:
                    break
            print("update times = " + str(update))
        return w
