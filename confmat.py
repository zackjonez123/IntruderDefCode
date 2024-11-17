"""
********** Confusion Matrix ********
"""
import RasPi
import nnet

def mat(res):
    f_count = 0
    h_count = 0
    if res == 0:
        f_count += 1
    else:
        h_count += 1
    f_total = 1 - ((100 - f_count) / 100)
    h_total = 1 - ((100 - h_count) / 100)
    result = [f_total, h_total]
    return result


def main():
    print("Confusion Matrix")
    print("Friendly case")
    friendly_path = "" #100 images of friendly occupant
    friendly_crop = ""
    nnet.cropAll(friendly_path, friendly_crop, "fcrop")
    res1 = RasPi.path(friendly_crop)
    final1 = mat(res1)
    print("Number of Friendly ID", final1[0], "Number of Hostile ID", final1[1])

    print("Hostile case")
    hostile_path = "" #100 images of hostile occupant
    hostile_crop = ""
    nnet.cropAll(hostile_path, hostile_crop, "hcrop")
    res2 = RasPi.path(hostile_crop)
    final2 = mat(res2)
    print("Number of Friendly ID", final2[0], "Number of Hostile ID", final2[1])



if __name__ == '__main__':
    main()
