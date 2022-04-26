from graphiler.mpdfg import split, reorder, fusion


def optimizer(mpdfg, opt_level):
    if opt_level == -1:
       split(mpdfg)
       return
    if opt_level == -2:
        split(mpdfg)
        reorder(mpdfg)
        return
    if opt_level == -3:
        split(mpdfg)
        reorder(mpdfg)
        fusion(mpdfg)
        return
    if opt_level == 0:
        return
    if opt_level > 0:
        split(mpdfg)
        reorder(mpdfg)
        reorder(mpdfg)
    if opt_level > 1:   
        # convergence check?
        for _ in range(0):
            reorder(mpdfg)
            fusion(mpdfg)
    return
