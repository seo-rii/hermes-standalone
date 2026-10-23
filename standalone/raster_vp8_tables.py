"""VP8 normative tables, from the Go Authors' BSD-licensed x/image/vp8.

Source commit a998c07120b3b6fbcb5f4b74d9037fd765fd25a9; see
vp8-provenance.json and licenses/go-image-LICENSE for source hashes and terms.
Copyright 2011 The Go Authors. All rights reserved.
"""

tokenProbUpdateProb = bytes.fromhex(
    'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffb0f6ffffffffffffffffffdff1fcffffff'
    'fffffffffff9fdfdfffffffffffffffffff4fcffffffffffffffffeafefefffffffffffffffffdffffffffffffffffffffff'
    'f6feffffffffffffffffeffdfefffffffffffffffffefffefffffffffffffffffff8fefffffffffffffffffbfffeffffffff'
    'fffffffffffffffffffffffffffffffffdfefffffffffffffffffbfefefffffffffffffffffefffefffffffffffffffffffe'
    'fdfffefffffffffffffafffefffefffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'
    'ffffffffffffffffffffffffffffd9ffffffffffffffffffffe1fcf1fdfffffeffffffffeafaf1fafdfffdfefffffffffeff'
    'ffffffffffffffffdffefeffffffffffffffffeefdfefefffffffffffffffff8fefffffffffffffffff9feffffffffffffff'
    'fffffffffffffffffffffffffffffdfffffffffffffffffff7fefffffffffffffffffffffffffffffffffffffffffffdfeff'
    'fffffffffffffffcfffffffffffffffffffffffffffffffffffffffffffffefefffffffffffffffffdffffffffffffffffff'
    'fffffffffffffffffffffffffffefdfffffffffffffffffafffffffffffffffffffffeffffffffffffffffffffffffffffff'
    'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffbafbfaffffffffffffffffeafbf4feffffffffffffff'
    'fbfbf3fdfefffefffffffffffdfeffffffffffffffffecfdfefffffffffffffffffbfdfdfefefffffffffffffffefeffffff'
    'fffffffffffefefefffffffffffffffffffffffffffffffffffffffffefffffffffffffffffffefefffffffffffffffffffe'
    'fffffffffffffffffffffffffffffffffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffffffffffff'
    'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'
    'fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff8ffffffffffffff'
    'fffffffafefcfefffffffffffffff8fef9fdfffffffffffffffffdfdfffffffffffffffff6fdfdfffffffffffffffffcfefb'
    'fefefffffffffffffffefcfffffffffffffffff8fefdfffffffffffffffffdfffefefffffffffffffffffbfeffffffffffff'
    'fffff5fbfefffffffffffffffffdfdfefffffffffffffffffffbfdfffffffffffffffffcfdfefffffffffffffffffffeffff'
    'fffffffffffffffffcfffffffffffffffffff9fffefffffffffffffffffffffefffffffffffffffffffffdffffffffffffff'
    'fffafffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffffffffffffffffffffffffff'
    'ffffffffffff'
)

defaultTokenProb = bytes.fromhex(
    '808080808080808080808080808080808080808080808080808080808080808080fd88feffe4db8080808080bd81f2ffe3d5'
    'ffdb8080806a7ee3fcd6d1ffff8080800162f8ffece2ffff808080b585eefeddeaff9a8080804e86caf7c6b4ffdb80808001'
    'b9f9fff3ff8080808080b896f7ffece080808080804d6ed8ffece680808080800165fbfff1ff8080808080aa8bf1fcecd1ff'
    'ff8080802574c4f3e4ffffff80808001ccfefff5ff8080808080cfa0faffee8080808080806667e7ffd3ab80808080800198'
    'fcfff0ff8080808080b187f3ffeae180808080805081d3ffc2e080808080800101ff8080808080808080f601ff8080808080'
    '808080ff80808080808080808080c623eddfc1bba2a0919b3e832dc6ddacb0dc9dfcdd01442f92d095a7dda2ffdf800195f1'
    'ffdde0ffff808080b88deafddedcffc78080805163b5f2b0bef9caffff800181e8fdd6c5f2c4ffff806379d2fac9c6ffca80'
    '8080175ba3f2aabbf7d2ffff8001c8f6ffeaff80808080806db2f1ffe7f5ffff8080802c82c9fdcdc0ffff8080800184effb'
    'dbd1ffa58080805e88e1fbdabeffff8080801664aef5baa1ffc780808001b6f9ffe8eb80808080807c8ff1ffe3ea80808080'
    '80234db5fbc1d3ffcd808080019df7ffece7ffff808080798debffe1e3ffff8080802d63bcfbc3d9ffe08080800101fbffd5'
    'ff8080808080cb01f8ffff8080808080808901b1ffe0ff8080808080fd09f8fbcfd0ffc0808080af0de0f3c1b9f9c6ffff80'
    '4911abdda1b3eca7ffea80015ff7fdd4b7ffff808080ef5af4fad3d1ffff8080809b4dc3f8bcc3ffff8080800118effbdadb'
    'ffcd808080c933dbffc4ba8080808080452ebeefc9daffe480808001bffbffff808080808080dfa5f9ffd5ff80808080808d'
    '7cf8ffff8080808080800110f8ffff808080808080be24e6ffecff80808080809501ff808080808080808001e2ff80808080'
    '80808080f7c0ff8080808080808080f080ff80808080808080800186fcffff808080808080d53efaffff808080808080375d'
    'ff8080808080808080808080808080808080808080808080808080808080808080808080808080808080ca18d5ebbabfdca0'
    'f0afff7e26b6e8a9b8e4aeffbb803d2e8adb97b2f0aaffd8800170e6fac7bff79fffff80a66de4fcd3d7ffae808080274da2'
    'e8acb4f5b2ffff800134dcf6c6c7f9dcffff807c4abff3b7c1faddffff80184782db9aaaf3b6ffff8001b6e1f9dbf0ffe080'
    '80809596e2fcd8cdffab8080801c6caaf2b7c2fedfffff800151e6fccccbffc08080807b66d1f7bcc4ffe9808080145f99f3'
    'a4adffcb80808001def8ffd8d58080808080a8aff6fcebcdffff8080802f74d7ffd3d4ffff8080800179ecfdd4d6ffff8080'
    '808d54d5fcc9caffdb8080802a50a0f0a2b9ffcd8080800101ff8080808080808080f401ff8080808080808080ee01ff8080'
    '808080808080'
)

predProb = bytes.fromhex(
    'e7783059737178987098b3407eaa762e465faf458f505552489b67383a0aabdabd110d98721a11a32cc3150aad791850c31a'
    '3e2c405590470a26abd590221aaa2e371388a021ce473f14087272d00c09e251280b60b6541d102486b7598962656aa59448'
    'bb64829d6f204b504266a7634a3e28ea80293509b2f18d1a086b4a2b1a9249a631179d412669a033341f7380684f0c1bd9ff'
    '5711075744472c72330fba172f290e6eb6b71511c2422d1966c5bd171216585893962a2e2dc4cd2b61b775552623b33d2735'
    'c8571a152be8ab3822336872661d5d4d271c55ab3aa55a6240221674ce17222ba6496b36201a3301512b1f44196a1640ab24'
    'e1722213156684bc104c7c3e124e5f5539323033c165239fd76f592e6f3c941facdbe415126f70714d55b3ff267872282a01'
    'c4f5d10a196d582b1d8ca6d5252b9a3d3f1e9b432d4401d16450082b9a01331a478e4e4e10ff8022c5ab29280566d3b70401'
    'dd333211a8d1c01719528a1f24ab1ba6262ce543573aa952731a3bb33f3b5ab43ba65d499a282815748fd12227af2f0f10b7'
    '22df312db72e1121b706620f20b7392e16188001361125412049731c801780cd2803097333c01206df572509733b4d40152f'
    '68372cda09363582e2405a46cd2829171a39363970b8052926a6d51e221a8598740a2086271335dd1a722049ff1f0941ea02'
    '0f0176494b200c33c0ffa02b33581f2343665537ba553815176f3bcd2d25c03726467c49660122627d622a58685575af525f'
    '543559806471652d4b4f7b2f338051ab0139110547663935293126210d7939491a0155290a438a4d6e5a2f727315020a66ff'
    'a61706651d100a558065c41a39120a6666d522142b75140f24a38044011a663d472522351ff3c0453c472649771cde25442d'
    '8022012f0bf5ab3e1113469255373e46252b259a64a355a0013f095c881c4020c9554b0f090940ffb8771056061c0540ff19'
    'f8013808118489ff3774803a0f145287391a7928a4321f899a851923da33672c83837b1f069e5628408794e02db780161a11'
    '83f09a0e01d12d10155b40de0701c53815279b3c8a1766d5530c0d36c0ff442f1c551a555580802092ab120b073f90ab0404'
    'f6231b0a92aeab0c1a80be502363b4507e362d557e2f57b033291420654b808b769274805538290fb0ec5525093e471e1177'
    '76ff11128a65263c8a37462b1a8e9224131eabff611b148a2d3d3edb0151bc4020291475978e1415a370130c3dc380300418'
)

dequantTableDC = (4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 17, 18, 19, 20, 20, 21, 21, 22, 22, 23, 23,
 24, 25, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
 71, 72, 73, 74, 75, 76, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 93, 95, 96, 98,
 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118, 122, 124, 126, 128, 130, 132, 134, 136, 138,
 140, 143, 145, 148, 151, 154, 157)

dequantTableAC = (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
 55, 56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98,
 100, 102, 104, 106, 108, 110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149,
 152, 155, 158, 161, 164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209, 213, 217, 221,
 225, 229, 234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284)

