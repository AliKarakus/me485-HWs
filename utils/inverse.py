# Bespoke code generator for LU decomposition & Forward/Backward substitution

def make_lu_dcmp(be, nvars):
    f_txt = "def _dcmp(A):\n"
    if nvars == 1:
        f_txt += "    A[0][0] = 1/A[0][0]"
        
    else:
        for i in range(1, nvars):
            f_txt += "    A[{}][0] /= A[0][0]\n".format(i)
            for j in range(1, nvars):
                if i<=j:
                    subtxt = "A[{}][0]*A[0][{}]".format(i,j)
                    for k in range(1, i):
                        subtxt += " + A[{}][{}]*A[{}][{}]".format(i,k,k,j)
                    f_txt += "    A[{}][{}] -= ".format(i,j)+subtxt+"\n"
                else:
                    subtxt = "    A[{}][{}] = (A[{}][{}]".format(i,j,i,j)
                    for k in range(j):
                        subtxt += " - A[{}][{}]*A[{}][{}]".format(i,k,k,j)
                    subtxt += ")/A[{}][{}]\n".format(j,j)
                    f_txt += subtxt
    
    gvar = {}
    lvar = {}
    exec(f_txt, gvar, lvar)
    
    return be.compile(lvar['_dcmp'])


def make_substitution(be, nvars):
    f_txt = "def _sub(A, b):\n"
    if nvars == 1:
        f_txt += "    b[0] *= A[0][0]"

    else:
        # Forward substitution
        for i in range(1, nvars):
            sub_txt = "    b[{}] -= A[{}][0]*b[0]".format(i,i)
            for k in range(1, i):
                sub_txt += " + A[{}][{}]*b[{}]".format(i,k,k)
            f_txt += sub_txt + "\n"

        # Backward substitution
        f_txt += "    b[{}] /= A[{}][{}]\n".format(nvars-1, nvars-1, nvars-1)
        for i in range(nvars-2, -1, -1):
            sub_txt = "    b[{}] = (b[{}]".format(i,i)
            for k in range(i+1, nvars):
                sub_txt += " - A[{}][{}]*b[{}]".format(i,k,k)
            sub_txt += ")/A[{}][{}]\n".format(i,i)
            f_txt += sub_txt

    gvar = {}
    lvar = {}
    exec(f_txt, gvar, lvar)

    return be.compile(lvar['_sub'])