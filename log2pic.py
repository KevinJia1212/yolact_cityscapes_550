import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import ast
import json


####### concat 2 logs of the same training process
def concat(file_1, file_2):
    output_file = "./logs/cityscapes_2020-01-12_550.log" 
    f1=open(file_1,"r")
    f2=open(file_2,"r")
    ff1 = f1.readlines()
    ff2 = f2.readlines()
    for line in ff1:
        with open(output_file, 'a') as file:
            file.write(line)
    for line in ff2:
        with open(output_file, 'a') as file:
            file.write(line)

# # for i in range(len(lines)):
# #     a = lines[0]
# #     line_dic = json.loads(lines[i]) 
# #     if line_dic['type'] == "session":
# #         start_points.append(i)

def filter(x, y, win_size):
    temp = 0
    out_x = []
    out_y = []
    for i in range(len(y)):
        if i % win_size != 0 or i == 0:
            temp += y[i]
        else:
            avg = temp / win_size
            temp = y[i]
            out_x.append(x[i-1])
            out_y.append(avg)
    return out_x, out_y
            
            
if __name__ == "__main__":
    print("Reading......")
    logfile = open("./logs/cityscapes_2020-01-12_550.log", "r")
    lines = logfile.readlines() 
    
    train_iter = []
    loss_box = []
    loss_mask = []
    loss_class = []
    loss_score = []
    val_iter = []
    box_mAP = []
    mask_mAP = []

    interval_train = 750
    interval_val = 3

    for i in range(len(lines)):
        line_dic = json.loads(lines[i]) 

        if line_dic['type'] == "train":
            train_iter.append(line_dic['data']['iter'])
            loss_box.append(line_dic['data']['loss']['B'])
            loss_mask.append(line_dic['data']['loss']['M'])
            loss_class.append(line_dic['data']['loss']['C'])
            loss_score.append(line_dic['data']['loss']['S'])

        elif line_dic['type'] == 'val':
            val_iter.append(line_dic['data']['iter'])
            box_mAP.append(line_dic['data']['box']['all'])
            mask_mAP.append(line_dic['data']['mask']['all'])
        
        else:
            continue
    print("log file imported!")

    x, b_loss = filter(train_iter, loss_box, interval_train)
    plt.subplot(4, 1, 1)
    plt.plot(x, b_loss, color="blue", linewidth=1)
    plt.ylabel("B-Box Loss")
    plt.title("Training Losses")
    x, c_loss = filter(train_iter, loss_class, interval_train)
    plt.subplot(4, 1, 2)
    plt.plot(x, c_loss, color="orange", linewidth=1)
    plt.ylabel("Class Loss")
    x, m_loss = filter(train_iter, loss_mask, interval_train)
    plt.subplot(4, 1, 3)
    plt.plot(x, m_loss, color="green", linewidth=1)
    plt.ylabel("Mask Loss")
    x, s_loss = filter(train_iter, loss_score, interval_train)
    plt.subplot(4, 1, 4)
    plt.plot(x, s_loss, color="pink", linewidth=1)
    plt.ylabel("Score Loss")
    plt.xlabel("Iteration")   
    

    loss_total = []
    for i in range(len(x)):
        total = b_loss[i] + m_loss[i] + c_loss[i] + s_loss[i]
        loss_total.append(total) 
    x, total = filter(x, loss_total, interval_val)
    plt.figure()
    plt.plot(x, total, color="red", linewidth=1)
    plt.title('Total Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Total Loss')

    plt.figure()
    x, box = filter(val_iter, box_mAP, interval_val)
    x, mask = filter(val_iter, mask_mAP, interval_val)
    plt.plot(x, box, label='B-Box mAP', color="blue", linewidth=1)
    plt.plot(x, mask, label='Mask mAP', color="red", linewidth=1)
    plt.title('Validation')
    plt.xlabel('Iteration')
    plt.ylabel('mAP')
    plt.legend()
    plt.show()

    logfile.close()