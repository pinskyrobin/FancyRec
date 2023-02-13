from matplotlib import pyplot as plt
import numpy as np


def show_results():
    loss = eval(open('../out/loss.txt', 'r').read())
    val_auc = eval(open('../out/val_auc.txt', 'r').read())
    test_auc = eval(open('../out/test_auc.txt', 'r').read())
    # print(len(loss))
    # print(loss)

    x = np.arange(0, len(loss))
    y = np.array(loss, dtype=np.float32)
    # draw loss
    plt.title("Train_loss")
    plt.xlabel("batch")
    plt.ylabel("value")
    plt.plot(x, y, label='loss', linestyle='-')
    plt.legend(loc='upper right')
    plt.show()

    # draw auc
    test_auc = list(zip(*test_auc))
    print(test_auc)
    plt.title("Test_AUC")
    x = np.arange(1, len(test_auc[0])+1)
    y1 = np.array(test_auc[0])
    y2 = np.array(test_auc[1])
    y3 = np.array(test_auc[2])
    plt.plot(x, y1, color='blue', label='AUC', linestyle='-')
    plt.plot(x, y2, color='green', label='NDCG_10', linestyle='-')
    plt.plot(x, y3, color='red', label='NDCG_50', linestyle='-')
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    show_results()
