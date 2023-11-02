
import os
import sys
import torch

__ed_dict__ = {}

def str_edit_distance(str_ref, str_hyp):

    global __ed_dict__
    if len(__ed_dict__) > 100000:   # NOTE: avoid the dictionary size to increase too much
        __ed_dict__ = {}

    rtoks = str_ref.split()
    htoks = str_hyp.split()
    for t in rtoks + htoks:
        if t not in __ed_dict__:
            __ed_dict__[t] = len(__ed_dict__)
    ref = torch.LongTensor( [__ed_dict__[t] for t in rtoks] )
    hyp = torch.LongTensor( [__ed_dict__[t] for t in htoks] )

    # These are 3, 3 and 4 in sclite if I remember well. However it does not change much the final error rate
    ins_weight = 1.0
    del_weight = 1.0
    sub_weight = 1.0

    curr_x_size = ref.size(0)
    curr_y_size = hyp.size(0)
    tsr_ed_matrix = torch.FloatTensor(curr_x_size+1, curr_y_size+1).fill_(0)
    for i in range(1, curr_x_size+1):
        tsr_ed_matrix[i,0] = float(i)
    for i in range(1, curr_y_size+1):
        tsr_ed_matrix[0,i] = float(i)

    for ij in range(curr_x_size*curr_y_size):
        i = (ij // curr_y_size)+1
        j = (ij % curr_y_size)+1
                            
        tmp_weight = 0
        if ref[i-1] != hyp[j-1]:
            tmp_weight = sub_weight 
        tsr_ed_matrix[i,j] = min(tsr_ed_matrix[i-1,j] + del_weight, tsr_ed_matrix[i,j-1] + ins_weight, tsr_ed_matrix[i-1,j-1] + tmp_weight)

    # Back-tracking for error rate computation
    n_ins = 0
    n_del = 0
    n_sub = 0

    alignement = []
    back_track_i = curr_x_size
    back_track_j = curr_y_size
    while back_track_i > 0 and back_track_j > 0:
        
        i = back_track_i
        j = back_track_j
        tmp_weight = 0
        if tsr_ed_matrix[i-1,j-1] != tsr_ed_matrix[i,j]:
            tmp_weight = sub_weight

        if tsr_ed_matrix[i-1,j] < tsr_ed_matrix[i,j-1]:
            if tsr_ed_matrix[i-1,j] < tsr_ed_matrix[i-1,j-1]:
                alignement.append( ('del', back_track_i-1, None) )
                n_del += 1
                back_track_i -= 1
            else:
                back_track_i -= 1
                back_track_j -= 1
                alignement.append( ('match', back_track_i, back_track_j) )
                if tmp_weight > 0:
                    n_sub += 1
                    alignement[-1] = ('sub', alignement[-1][1], alignement[-1][2])

        else:   # tsr_ed_matrix[i-1,j] >= tsr_ed_matrix[i,j-1]
            if tsr_ed_matrix[i,j-1] < tsr_ed_matrix[i-1,j-1]:
                alignement.append( ('ins', back_track_i-1, back_track_j-1) )
                n_ins += 1
                back_track_j -= 1
            else:
                back_track_i -= 1
                back_track_j -= 1
                alignement.append( ('match', back_track_i, back_track_j) )
                if tmp_weight > 0:
                    n_sub += 1
                    alignement[-1] = ('sub', alignement[-1][1], alignement[-1][2])

    while back_track_i > 0:
        alignement.append( ('del', back_track_i-1, None) )
        back_track_i -= 1
        n_del += 1

    while back_track_j > 0:
        alignement.append( ('ins', back_track_i-1, back_track_j-1) )
        back_track_j -= 1
        n_ins += 1

    alignement.reverse()
    return (n_ins, n_del, n_sub, curr_x_size, alignement)


def main(args):

    ref_str = args[1]
    hyp_str = args[2]

    print(' * Computing edit distance between:')
    print(' * ref: {}'.format(ref_str))
    print(' * hyp: {}'.format(hyp_str))
    print(' ---')

    er_vals = str_edit_distance(ref_str, hyp_str)

    print(' * ER: {:.2f}'.format(sum(er_vals[:3])/er_vals[3]))
    print(' * Errors:')
    print(' * ins: {}'.format(er_vals[0]))
    print(' * del: {}'.format(er_vals[1]))
    print(' * sub: {}'.format(er_vals[2]))
    print(' ---')

    rtoks = ref_str.split()
    htoks = hyp_str.split()
    alignement = er_vals[4]
    print( '* Alignement:' )
    for t in alignement:
        print(' * {}) r:{}, h:{}'.format(t[0], rtoks[t[1]], htoks[t[2]] if t[2] is not None else '-'))
    print(' ---')


if __name__ == '__main__':
    main(sys.argv)

