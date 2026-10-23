"""VP8 spatial predictors translated from x/image/vp8/predfunc.go.

Copyright 2011 The Go Authors. All rights reserved.
BSD-3-Clause: see licenses/go-image-LICENSE and vp8-provenance.json.
"""

def clip(value, minimum, maximum):
    return max(minimum, min(maximum, value))

def predFunc4DC(z, y, x):
    sum = int(4)
    for i in range(0, 4):
        sum += int(z.ybr[y-1][x+i])
    for j in range(0, 4):
        sum += int(z.ybr[y+j][x-1])
    avg = int(sum // 8)
    for j in range(0, 4):
        for i in range(0, 4):
            z.ybr[y+j][x+i] = avg

def predFunc4TM(z, y, x):
    delta0 = -int(z.ybr[y-1][x-1])
    for j in range(0, 4):
        delta1 = delta0 + int(z.ybr[y+j][x-1])
        for i in range(0, 4):
            delta2 = delta1 + int(z.ybr[y-1][x+i])
            z.ybr[y+j][x+i] = int(clip(delta2, 0, 255))

def predFunc4VE(z, y, x):
    a = int(z.ybr[y-1][x-1])
    b = int(z.ybr[y-1][x+0])
    c = int(z.ybr[y-1][x+1])
    d = int(z.ybr[y-1][x+2])
    e = int(z.ybr[y-1][x+3])
    f = int(z.ybr[y-1][x+4])
    abc = int((a + 2*b + c + 2) // 4)
    bcd = int((b + 2*c + d + 2) // 4)
    cde = int((c + 2*d + e + 2) // 4)
    smooth_def = int((d + 2*e + f + 2) // 4)
    for j in range(0, 4):
        z.ybr[y+j][x+0] = abc
        z.ybr[y+j][x+1] = bcd
        z.ybr[y+j][x+2] = cde
        z.ybr[y+j][x+3] = smooth_def

def predFunc4HE(z, y, x):
    s = int(z.ybr[y+3][x-1])
    r = int(z.ybr[y+2][x-1])
    q = int(z.ybr[y+1][x-1])
    p = int(z.ybr[y+0][x-1])
    a = int(z.ybr[y-1][x-1])
    ssr = int((s + 2*s + r + 2) // 4)
    srq = int((s + 2*r + q + 2) // 4)
    rqp = int((r + 2*q + p + 2) // 4)
    apq = int((a + 2*p + q + 2) // 4)
    for i in range(0, 4):
        z.ybr[y+0][x+i] = apq
        z.ybr[y+1][x+i] = rqp
        z.ybr[y+2][x+i] = srq
        z.ybr[y+3][x+i] = ssr

def predFunc4RD(z, y, x):
    s = int(z.ybr[y+3][x-1])
    r = int(z.ybr[y+2][x-1])
    q = int(z.ybr[y+1][x-1])
    p = int(z.ybr[y+0][x-1])
    a = int(z.ybr[y-1][x-1])
    b = int(z.ybr[y-1][x+0])
    c = int(z.ybr[y-1][x+1])
    d = int(z.ybr[y-1][x+2])
    e = int(z.ybr[y-1][x+3])
    srq = int((s + 2*r + q + 2) // 4)
    rqp = int((r + 2*q + p + 2) // 4)
    qpa = int((q + 2*p + a + 2) // 4)
    pab = int((p + 2*a + b + 2) // 4)
    abc = int((a + 2*b + c + 2) // 4)
    bcd = int((b + 2*c + d + 2) // 4)
    cde = int((c + 2*d + e + 2) // 4)
    z.ybr[y+0][x+0] = pab
    z.ybr[y+0][x+1] = abc
    z.ybr[y+0][x+2] = bcd
    z.ybr[y+0][x+3] = cde
    z.ybr[y+1][x+0] = qpa
    z.ybr[y+1][x+1] = pab
    z.ybr[y+1][x+2] = abc
    z.ybr[y+1][x+3] = bcd
    z.ybr[y+2][x+0] = rqp
    z.ybr[y+2][x+1] = qpa
    z.ybr[y+2][x+2] = pab
    z.ybr[y+2][x+3] = abc
    z.ybr[y+3][x+0] = srq
    z.ybr[y+3][x+1] = rqp
    z.ybr[y+3][x+2] = qpa
    z.ybr[y+3][x+3] = pab

def predFunc4VR(z, y, x):
    r = int(z.ybr[y+2][x-1])
    q = int(z.ybr[y+1][x-1])
    p = int(z.ybr[y+0][x-1])
    a = int(z.ybr[y-1][x-1])
    b = int(z.ybr[y-1][x+0])
    c = int(z.ybr[y-1][x+1])
    d = int(z.ybr[y-1][x+2])
    e = int(z.ybr[y-1][x+3])
    ab = int((a + b + 1) // 2)
    bc = int((b + c + 1) // 2)
    cd = int((c + d + 1) // 2)
    de = int((d + e + 1) // 2)
    rqp = int((r + 2*q + p + 2) // 4)
    qpa = int((q + 2*p + a + 2) // 4)
    pab = int((p + 2*a + b + 2) // 4)
    abc = int((a + 2*b + c + 2) // 4)
    bcd = int((b + 2*c + d + 2) // 4)
    cde = int((c + 2*d + e + 2) // 4)
    z.ybr[y+0][x+0] = ab
    z.ybr[y+0][x+1] = bc
    z.ybr[y+0][x+2] = cd
    z.ybr[y+0][x+3] = de
    z.ybr[y+1][x+0] = pab
    z.ybr[y+1][x+1] = abc
    z.ybr[y+1][x+2] = bcd
    z.ybr[y+1][x+3] = cde
    z.ybr[y+2][x+0] = qpa
    z.ybr[y+2][x+1] = ab
    z.ybr[y+2][x+2] = bc
    z.ybr[y+2][x+3] = cd
    z.ybr[y+3][x+0] = rqp
    z.ybr[y+3][x+1] = pab
    z.ybr[y+3][x+2] = abc
    z.ybr[y+3][x+3] = bcd

def predFunc4LD(z, y, x):
    a = int(z.ybr[y-1][x+0])
    b = int(z.ybr[y-1][x+1])
    c = int(z.ybr[y-1][x+2])
    d = int(z.ybr[y-1][x+3])
    e = int(z.ybr[y-1][x+4])
    f = int(z.ybr[y-1][x+5])
    g = int(z.ybr[y-1][x+6])
    h = int(z.ybr[y-1][x+7])
    abc = int((a + 2*b + c + 2) // 4)
    bcd = int((b + 2*c + d + 2) // 4)
    cde = int((c + 2*d + e + 2) // 4)
    smooth_def = int((d + 2*e + f + 2) // 4)
    efg = int((e + 2*f + g + 2) // 4)
    fgh = int((f + 2*g + h + 2) // 4)
    ghh = int((g + 2*h + h + 2) // 4)
    z.ybr[y+0][x+0] = abc
    z.ybr[y+0][x+1] = bcd
    z.ybr[y+0][x+2] = cde
    z.ybr[y+0][x+3] = smooth_def
    z.ybr[y+1][x+0] = bcd
    z.ybr[y+1][x+1] = cde
    z.ybr[y+1][x+2] = smooth_def
    z.ybr[y+1][x+3] = efg
    z.ybr[y+2][x+0] = cde
    z.ybr[y+2][x+1] = smooth_def
    z.ybr[y+2][x+2] = efg
    z.ybr[y+2][x+3] = fgh
    z.ybr[y+3][x+0] = smooth_def
    z.ybr[y+3][x+1] = efg
    z.ybr[y+3][x+2] = fgh
    z.ybr[y+3][x+3] = ghh

def predFunc4VL(z, y, x):
    a = int(z.ybr[y-1][x+0])
    b = int(z.ybr[y-1][x+1])
    c = int(z.ybr[y-1][x+2])
    d = int(z.ybr[y-1][x+3])
    e = int(z.ybr[y-1][x+4])
    f = int(z.ybr[y-1][x+5])
    g = int(z.ybr[y-1][x+6])
    h = int(z.ybr[y-1][x+7])
    ab = int((a + b + 1) // 2)
    bc = int((b + c + 1) // 2)
    cd = int((c + d + 1) // 2)
    de = int((d + e + 1) // 2)
    abc = int((a + 2*b + c + 2) // 4)
    bcd = int((b + 2*c + d + 2) // 4)
    cde = int((c + 2*d + e + 2) // 4)
    smooth_def = int((d + 2*e + f + 2) // 4)
    efg = int((e + 2*f + g + 2) // 4)
    fgh = int((f + 2*g + h + 2) // 4)
    z.ybr[y+0][x+0] = ab
    z.ybr[y+0][x+1] = bc
    z.ybr[y+0][x+2] = cd
    z.ybr[y+0][x+3] = de
    z.ybr[y+1][x+0] = abc
    z.ybr[y+1][x+1] = bcd
    z.ybr[y+1][x+2] = cde
    z.ybr[y+1][x+3] = smooth_def
    z.ybr[y+2][x+0] = bc
    z.ybr[y+2][x+1] = cd
    z.ybr[y+2][x+2] = de
    z.ybr[y+2][x+3] = efg
    z.ybr[y+3][x+0] = bcd
    z.ybr[y+3][x+1] = cde
    z.ybr[y+3][x+2] = smooth_def
    z.ybr[y+3][x+3] = fgh

def predFunc4HD(z, y, x):
    s = int(z.ybr[y+3][x-1])
    r = int(z.ybr[y+2][x-1])
    q = int(z.ybr[y+1][x-1])
    p = int(z.ybr[y+0][x-1])
    a = int(z.ybr[y-1][x-1])
    b = int(z.ybr[y-1][x+0])
    c = int(z.ybr[y-1][x+1])
    d = int(z.ybr[y-1][x+2])
    sr = int((s + r + 1) // 2)
    rq = int((r + q + 1) // 2)
    qp = int((q + p + 1) // 2)
    pa = int((p + a + 1) // 2)
    srq = int((s + 2*r + q + 2) // 4)
    rqp = int((r + 2*q + p + 2) // 4)
    qpa = int((q + 2*p + a + 2) // 4)
    pab = int((p + 2*a + b + 2) // 4)
    abc = int((a + 2*b + c + 2) // 4)
    bcd = int((b + 2*c + d + 2) // 4)
    z.ybr[y+0][x+0] = pa
    z.ybr[y+0][x+1] = pab
    z.ybr[y+0][x+2] = abc
    z.ybr[y+0][x+3] = bcd
    z.ybr[y+1][x+0] = qp
    z.ybr[y+1][x+1] = qpa
    z.ybr[y+1][x+2] = pa
    z.ybr[y+1][x+3] = pab
    z.ybr[y+2][x+0] = rq
    z.ybr[y+2][x+1] = rqp
    z.ybr[y+2][x+2] = qp
    z.ybr[y+2][x+3] = qpa
    z.ybr[y+3][x+0] = sr
    z.ybr[y+3][x+1] = srq
    z.ybr[y+3][x+2] = rq
    z.ybr[y+3][x+3] = rqp

def predFunc4HU(z, y, x):
    s = int(z.ybr[y+3][x-1])
    r = int(z.ybr[y+2][x-1])
    q = int(z.ybr[y+1][x-1])
    p = int(z.ybr[y+0][x-1])
    pq = int((p + q + 1) // 2)
    qr = int((q + r + 1) // 2)
    rs = int((r + s + 1) // 2)
    pqr = int((p + 2*q + r + 2) // 4)
    qrs = int((q + 2*r + s + 2) // 4)
    rss = int((r + 2*s + s + 2) // 4)
    sss = int(s)
    z.ybr[y+0][x+0] = pq
    z.ybr[y+0][x+1] = pqr
    z.ybr[y+0][x+2] = qr
    z.ybr[y+0][x+3] = qrs
    z.ybr[y+1][x+0] = qr
    z.ybr[y+1][x+1] = qrs
    z.ybr[y+1][x+2] = rs
    z.ybr[y+1][x+3] = rss
    z.ybr[y+2][x+0] = rs
    z.ybr[y+2][x+1] = rss
    z.ybr[y+2][x+2] = sss
    z.ybr[y+2][x+3] = sss
    z.ybr[y+3][x+0] = sss
    z.ybr[y+3][x+1] = sss
    z.ybr[y+3][x+2] = sss
    z.ybr[y+3][x+3] = sss

def predFunc8DC(z, y, x):
    sum = int(8)
    for i in range(0, 8):
        sum += int(z.ybr[y-1][x+i])
    for j in range(0, 8):
        sum += int(z.ybr[y+j][x-1])
    avg = int(sum // 16)
    for j in range(0, 8):
        for i in range(0, 8):
            z.ybr[y+j][x+i] = avg

def predFunc8TM(z, y, x):
    delta0 = -int(z.ybr[y-1][x-1])
    for j in range(0, 8):
        delta1 = delta0 + int(z.ybr[y+j][x-1])
        for i in range(0, 8):
            delta2 = delta1 + int(z.ybr[y-1][x+i])
            z.ybr[y+j][x+i] = int(clip(delta2, 0, 255))

def predFunc8VE(z, y, x):
    for j in range(0, 8):
        for i in range(0, 8):
            z.ybr[y+j][x+i] = z.ybr[y-1][x+i]

def predFunc8HE(z, y, x):
    for j in range(0, 8):
        for i in range(0, 8):
            z.ybr[y+j][x+i] = z.ybr[y+j][x-1]

def predFunc8DCTop(z, y, x):
    sum = int(4)
    for j in range(0, 8):
        sum += int(z.ybr[y+j][x-1])
    avg = int(sum // 8)
    for j in range(0, 8):
        for i in range(0, 8):
            z.ybr[y+j][x+i] = avg

def predFunc8DCLeft(z, y, x):
    sum = int(4)
    for i in range(0, 8):
        sum += int(z.ybr[y-1][x+i])
    avg = int(sum // 8)
    for j in range(0, 8):
        for i in range(0, 8):
            z.ybr[y+j][x+i] = avg

def predFunc8DCTopLeft(z, y, x):
    for j in range(0, 8):
        for i in range(0, 8):
            z.ybr[y+j][x+i] = 0x80

def predFunc16DC(z, y, x):
    sum = int(16)
    for i in range(0, 16):
        sum += int(z.ybr[y-1][x+i])
    for j in range(0, 16):
        sum += int(z.ybr[y+j][x-1])
    avg = int(sum // 32)
    for j in range(0, 16):
        for i in range(0, 16):
            z.ybr[y+j][x+i] = avg

def predFunc16TM(z, y, x):
    delta0 = -int(z.ybr[y-1][x-1])
    for j in range(0, 16):
        delta1 = delta0 + int(z.ybr[y+j][x-1])
        for i in range(0, 16):
            delta2 = delta1 + int(z.ybr[y-1][x+i])
            z.ybr[y+j][x+i] = int(clip(delta2, 0, 255))

def predFunc16VE(z, y, x):
    for j in range(0, 16):
        for i in range(0, 16):
            z.ybr[y+j][x+i] = z.ybr[y-1][x+i]

def predFunc16HE(z, y, x):
    for j in range(0, 16):
        for i in range(0, 16):
            z.ybr[y+j][x+i] = z.ybr[y+j][x-1]

def predFunc16DCTop(z, y, x):
    sum = int(8)
    for j in range(0, 16):
        sum += int(z.ybr[y+j][x-1])
    avg = int(sum // 16)
    for j in range(0, 16):
        for i in range(0, 16):
            z.ybr[y+j][x+i] = avg

def predFunc16DCLeft(z, y, x):
    sum = int(8)
    for i in range(0, 16):
        sum += int(z.ybr[y-1][x+i])
    avg = int(sum // 16)
    for j in range(0, 16):
        for i in range(0, 16):
            z.ybr[y+j][x+i] = avg

def predFunc16DCTopLeft(z, y, x):
    for j in range(0, 16):
        for i in range(0, 16):
            z.ybr[y+j][x+i] = 0x80

predFunc4 = (predFunc4DC, predFunc4TM, predFunc4VE, predFunc4HE, predFunc4RD, predFunc4VR, predFunc4LD, predFunc4VL, predFunc4HD, predFunc4HU, None, None, None)
predFunc8 = (predFunc8DC, predFunc8TM, predFunc8VE, predFunc8HE, None, None, None, None, None, None, predFunc8DCTop, predFunc8DCLeft, predFunc8DCTopLeft)
predFunc16 = (predFunc16DC, predFunc16TM, predFunc16VE, predFunc16HE, None, None, None, None, None, None, predFunc16DCTop, predFunc16DCLeft, predFunc16DCTopLeft)
