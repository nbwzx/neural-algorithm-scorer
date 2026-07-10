# MIT License

# Copyright (c) 2021 trangium

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def alg_speed(sequence, ignoreErrors=False, ignoreauf=False,
              wristMult=0.8, pushMult=1.3, ringMult=1.4,
              destabilize=0.5, addRegrip=1, double=1.65,
              sesliceMult=1.25, overWorkMult=2.25,
              moveblock=0.8, rotation=3.5):
    """
    Returns estimated execution time (in arbitrary units) for a Rubik's cube algorithm.
    Lower is better. Translation of the JavaScript function.
    """

    def test(splitSeq, lGrip, rGrip, speed):
        def overwork(finger, locationPrefer, penalty=overWorkMult):
            if finger[1] != locationPrefer:
                if speed - finger[0] < penalty:
                    return penalty - speed + finger[0]
            return 0

        lThumb = [-1, "home"]
        lIndex = [-1, "home"]
        lMiddle = [-1, "home"]
        lRing = [-1, "home"]
        rThumb = [-1, "home"]
        rIndex = [-1, "home"]
        rMiddle = [-1, "home"]
        rRing = [-1, "home"]
        lOhCool = -1
        rOhCool = -1
        lWrist = lGrip
        rWrist = rGrip
        grip = 1
        udgrip = -1
        prevSpeed = None
        firstMoveSpeed = None

        for j, move in enumerate(splitSeq):
            normalMove = move.upper()
            prevMove = splitSeq[j-1].upper() if j > 0 else " "
            if prevSpeed is not None:
                firstMoveSpeed = speed
                speed = prevSpeed
            if j < len(splitSeq) - 1:
                if (move[0] == "U" and splitSeq[j+1][0] == "D") or (move[0] == "D" and splitSeq[j+1][0] == "U"):
                    prevSpeed = speed

            # ====== Move cases ======
            if normalMove == "R'":
                if rWrist == 2:
                    rWrist = 0
                elif rWrist > -1 and not (lWrist >= 1 and rWrist <= 0):
                    rWrist -= 1
                else:
                    return [j, speed, lWrist, rWrist-1,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]
                speed += wristMult

            elif normalMove == "R":
                if rWrist < 2 and not (lWrist <= -1 and rWrist >= 0):
                    rWrist += 1
                else:
                    return [j, speed, lWrist, rWrist+1,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]
                speed += wristMult

            elif normalMove == "R2":
                if rWrist >= 1 and lWrist < 1:
                    rWrist = -1
                elif lWrist > -1:
                    rWrist += 2
                else:
                    return [j, speed, lWrist, (rWrist-2 if rWrist > 0 else rWrist+2),
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]
                speed += double * wristMult

            elif normalMove == "U":
                if rWrist == 0 and (rThumb[0] + overWorkMult <= speed or rThumb[1] != "top") and rIndex[1] != "m":
                    if overwork(rIndex, "home") <= overwork(rMiddle, "home"):
                        speed += overwork(rIndex, "home")
                        speed += 1
                        rIndex = [speed, "uflick"]
                    else:
                        speed += overwork(rMiddle, "home")
                        speed += 1
                        rIndex = [speed, "uflick"]
                        rMiddle = [speed, "uflick"]
                elif rWrist == 1 and lWrist == 0:
                    speed += overwork(lIndex, "uflick")
                    if prevMove == "B'":
                        speed += moveblock + pushMult
                    elif prevMove[0] == "B'":
                        speed += moveblock * 0.5 + pushMult
                    else:
                        speed += pushMult
                    lIndex = [speed, "home"]
                elif lWrist == 0 and prevMove[0] not in ["F", "B"]:
                    if lIndex[1] == "uflick":
                        speed += overwork(lIndex, "eido", 0.75 * overWorkMult)
                        speed = max(speed, lOhCool + 2.5)
                    else:
                        speed += overwork(lIndex, "eido", 1.25 * overWorkMult)
                    speed += 1.15 * pushMult
                    lIndex = [speed, "uflick"]
                    lOhCool = speed
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "U'":
                if lWrist == 0 and (lThumb[0] + overWorkMult <= speed or lThumb[1] != "top") and lIndex[1] != "m":
                    if overwork(lIndex, "home") <= overwork(lMiddle, "home"):
                        speed += overwork(lIndex, "home")
                        speed += 1
                        lIndex = [speed, "uflick"]
                    else:
                        speed += overwork(lMiddle, "home")
                        speed += 1
                        lIndex = [speed, "uflick"]
                        lMiddle = [speed, "uflick"]
                elif lWrist == 1 and rWrist == 0:
                    speed += overwork(rIndex, "uflick")
                    if prevMove == "B":
                        speed += moveblock + pushMult
                    elif prevMove[0] == "B'":
                        speed += moveblock * 0.5 + pushMult
                    else:
                        speed += pushMult
                    rIndex = [speed, "home"]
                elif rWrist == 0 and prevMove[0] not in ["F", "B"]:
                    if rIndex[1] == "uflick":
                        speed += overwork(rIndex, "eido", 0.75 * overWorkMult)
                        speed = max(speed, rOhCool + 2.5)
                    else:
                        speed += overwork(rIndex, "eido", 1.25 * overWorkMult)
                    speed += 1.15 * pushMult
                    rIndex = [speed, "uflick"]
                    rOhCool = speed
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "U2":
                if rWrist == 0 and (lIndex[1] == "m" or lWrist != 0 or
                    max(overwork(rIndex, "home"), overwork(rMiddle, "home"), overwork(rRing, "u2grip", moveblock*overWorkMult)) <=
                    max(overwork(lIndex, "home"), overwork(lMiddle, "home"), overwork(lRing, "u2grip", moveblock*overWorkMult))):
                    speed += overwork(rIndex, "home")
                    speed += overwork(rMiddle, "home")
                    speed += overwork(rRing, "u2grip", moveblock * overWorkMult)
                    speed += double
                    rIndex = [speed, "uflick"]
                    rMiddle = [speed, "uflick"]
                elif lWrist == 0:
                    speed += overwork(lIndex, "home")
                    speed += overwork(lMiddle, "home")
                    speed += overwork(lRing, "u2grip", moveblock * overWorkMult)
                    speed += double
                    lIndex = [speed, "uflick"]
                    lMiddle = [speed, "uflick"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "D":
                if lWrist == 0 and (rWrist != 0 or
                    max(overwork(lRing, "home"), overwork(lMiddle, "home")) <=
                    max(overwork(rRing, "dflick"), overwork(rMiddle, "home"))):
                    speed += overwork(lRing, "home")
                    speed += overwork(lMiddle, "home")
                    if prevMove[0] == "B":
                        speed += moveblock * 0.5 + ringMult
                    else:
                        speed += ringMult
                    lRing = [speed, "dflick"]
                elif rWrist == 0 and prevMove[0] != "B":
                    speed += overwork(rRing, "dflick")
                    speed += overwork(rMiddle, "home")
                    speed += ringMult * pushMult
                    rRing = [speed, "home"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "D'":
                if rWrist == 0 and (lWrist != 0 or
                    max(overwork(rRing, "home"), overwork(rMiddle, "home")) <=
                    max(overwork(lRing, "dflick"), overwork(lMiddle, "home"))):
                    speed += overwork(rRing, "home")
                    speed += overwork(rMiddle, "home")
                    if prevMove[0] == "B":
                        speed += moveblock * 0.5 + ringMult
                    else:
                        speed += ringMult
                    rRing = [speed, "dflick"]
                elif lWrist == 0 and prevMove[0] != "B":
                    speed += overwork(lRing, "dflick")
                    speed += overwork(lMiddle, "home")
                    speed += ringMult * pushMult
                    lRing = [speed, "home"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "D2":
                if rWrist == 0 and (lWrist != 0 or
                    max(overwork(rMiddle, "home"), overwork(rRing, "home")) <=
                    max(overwork(lMiddle, "home"), overwork(lRing, "home"))):
                    speed += overwork(rMiddle, "home")
                    speed += overwork(rRing, "home")
                    if prevMove[0] == "B":
                        speed += moveblock * 0.5 + double * ringMult
                    else:
                        speed += double * ringMult
                    rRing = [speed, "dflick"]
                elif lWrist == 0:
                    speed += overwork(lMiddle, "home")
                    speed += overwork(lRing, "home")
                    if prevMove[0] == "B":
                        speed += moveblock * 0.5 + double * ringMult
                    else:
                        speed += double * ringMult
                    lRing = [speed, "dflick"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "F":
                if rWrist == -1:
                    speed += overwork(rIndex, "home")
                    speed += 1
                    rIndex = [speed, "uflick"]
                elif lWrist == 1 and move != "f":
                    speed += overwork(lRing, "home")
                    if prevMove[0] == "D":
                        speed += moveblock * 0.5 + ringMult
                    else:
                        speed += 1
                    lRing = [speed, "dflick"]
                elif rWrist == 1 and prevMove[0] != "D" and move != "f":
                    speed += overwork(rRing, "dflick")
                    speed += ringMult * pushMult
                    rRing = [speed, "home"]
                elif lWrist == -1 and rWrist == 0 and overwork(rIndex, "uflick") == 0:
                    speed += 1
                    rIndex = [speed, "fflick"]
                elif lWrist == -1 and overwork(lIndex, "uflick") == 0 and prevMove[0] != "U":
                    speed += pushMult
                    lIndex = [speed, "home"]
                elif lWrist == -1 and grip == -1:
                    speed += overwork(lThumb, "top", 0.9 * overWorkMult)
                    speed += overwork(lIndex, "top")
                    if prevMove[0] == "D":
                        speed += 1.8
                    else:
                        speed += 1
                    lWrist += 1
                    lThumb = [speed, "leftu"]
                    lIndex = [speed, "top"]
                elif lWrist == 0 and grip == -1:
                    speed += overwork(lThumb, "bottom")
                    speed += overwork(lIndex, "top")
                    if prevMove[0] == "D":
                        speed += 2.05
                    else:
                        speed += 1.25
                    lThumb = [speed, "top"]
                    lIndex = [speed, "top"]
                elif rWrist == 0 and lWrist == 0 and move == "f":
                    speed += overwork(rIndex, "uflick")
                    speed += overwork(rMiddle, "home")
                    speed += 1
                    rIndex = [speed, "fflick"]
                elif j == 0 and rWrist == 0 and lWrist == 0:
                    speed += overwork(rThumb, "top")
                    speed += 1
                    rThumb = [speed, "rdown"]
                    rMiddle = [speed, "uflick"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "F'":
                if lWrist == -1:
                    speed += overwork(lIndex, "home")
                    speed += 1
                    lIndex = [speed, "uflick"]
                elif rWrist == 1 and move != "f":
                    speed += overwork(rRing, "home")
                    if prevMove[0] == "D":
                        speed += moveblock * 0.5 + ringMult
                    else:
                        speed += 1
                    rRing = [speed, "dflick"]
                elif lWrist == 1 and prevMove[0] != "D" and move != "f":
                    speed += overwork(lRing, "dflick")
                    speed += ringMult * pushMult
                    lRing = [speed, "home"]
                elif rWrist == -1 and lWrist == 0 and overwork(lIndex, "uflick") == 0:
                    speed += 1
                    lIndex = [speed, "fflick"]
                elif rWrist == -1 and overwork(rIndex, "uflick") == 0 and prevMove[0] != "U":
                    speed += pushMult
                    rIndex = [speed, "home"]
                elif rWrist == -1 and grip == 1:
                    speed += overwork(rThumb, "top", 0.9 * overWorkMult)
                    speed += overwork(rIndex, "top")
                    if prevMove[0] == "D":
                        speed += 1.8
                    else:
                        speed += 1
                    rWrist += 1
                    rThumb = [speed, "rightu"]
                    rIndex = [speed, "top"]
                elif rWrist == 0 and grip == 1:
                    speed += overwork(rThumb, "bottom")
                    speed += overwork(rIndex, "top")
                    if prevMove[0] == "D":
                        speed += 2.05
                    else:
                        speed += 1.25
                    rThumb = [speed, "top"]
                    rIndex = [speed, "top"]
                elif lWrist == 0 and rWrist == 0 and move == "f'":
                    speed += overwork(lIndex, "uflick")
                    speed += overwork(lMiddle, "home")
                    speed += 1
                    lIndex = [speed, "fflick"]
                elif j == 0 and rWrist == 0 and lWrist == 0:
                    speed += overwork(lThumb, "top")
                    speed += 1
                    lThumb = [speed, "rdown"]
                    lMiddle = [speed, "uflick"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "F2":
                if rWrist == -1 and (lWrist != -1 or
                    max(overwork(rIndex, "home"), overwork(rMiddle, "home"), overwork(rRing, "u2grip", moveblock*overWorkMult)) <=
                    max(overwork(lIndex, "home"), overwork(lMiddle, "home"), overwork(lRing, "u2grip", moveblock*overWorkMult))):
                    speed += overwork(rIndex, "home")
                    speed += overwork(rMiddle, "home")
                    speed += overwork(rRing, "u2grip", moveblock * overWorkMult)
                    speed += double
                    rIndex = [speed, "uflick"]
                    rMiddle = [speed, "uflick"]
                elif lWrist == -1:
                    speed += overwork(lIndex, "home")
                    speed += overwork(lMiddle, "home")
                    speed += overwork(lRing, "u2grip", moveblock * overWorkMult)
                    speed += double
                    lIndex = [speed, "uflick"]
                    lMiddle = [speed, "uflick"]
                elif rWrist == 1 and (lWrist != 1 or
                    max(overwork(rMiddle, "home"), overwork(rRing, "home")) <=
                    max(overwork(lMiddle, "home"), overwork(lRing, "home"))):
                    speed += overwork(rMiddle, "home")
                    speed += overwork(rRing, "home")
                    if prevMove[0] == "D":
                        speed += double * ringMult + moveblock * 0.5
                    else:
                        speed += double * ringMult
                    rRing = [speed, "dflick"]
                elif lWrist == 1:
                    speed += overwork(lMiddle, "home")
                    speed += overwork(lRing, "home")
                    if prevMove[0] == "D":
                        speed += double * ringMult + moveblock * 0.5
                    else:
                        speed += double * ringMult
                    lRing = [speed, "dflick"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "L":
                if lWrist == 2:
                    lWrist = 0
                elif lWrist > -1 and not (rWrist >= 1 and lWrist <= 0):
                    lWrist -= 1
                else:
                    return [j, speed, lWrist-1, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]
                speed += wristMult

            elif normalMove == "L'":
                if lWrist < 2 and not (rWrist <= -1 and lWrist >= 0):
                    lWrist += 1
                else:
                    return [j, speed, lWrist+1, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]
                speed += wristMult

            elif normalMove == "L2":
                if lWrist >= 1 and rWrist < 1:
                    lWrist = -1
                elif rWrist > -1:
                    lWrist += 2
                else:
                    return [j, speed, (lWrist-2 if lWrist > 0 else lWrist+2), rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]
                speed += double * wristMult

            elif normalMove == "B":
                if rWrist == 1:
                    speed += overwork(rIndex, "home")
                    speed += 1
                    rIndex = [speed, "uflick"]
                elif lWrist == -1:
                    speed += overwork(lRing, "home")
                    speed += overwork(lMiddle, "home")
                    if prevMove[0] == "U":
                        speed += moveblock * 0.5 + ringMult
                    else:
                        speed += ringMult
                    lRing = [speed, "dflick"]
                elif lWrist == 1 and prevMove[0] not in ["U", "D"]:
                    if lIndex[1] == "uflick":
                        speed += overwork(lIndex, "eido", 0.75 * overWorkMult)
                        speed = max(speed, lOhCool + 2.5)
                    else:
                        speed += overwork(lIndex, "eido", 1.25 * overWorkMult)
                    speed += 1.15 * pushMult
                    lIndex = [speed, "uflick"]
                    lOhCool = speed
                elif lWrist == 0 and (rWrist == 1 or rWrist == -1):
                    speed += overwork(lIndex, "top", 0.9 * overWorkMult)
                    if prevMove[0] == "U":
                        speed += 1.45
                    else:
                        speed += 1
                    lIndex = [speed, "leftdb"]
                elif rWrist == -1 and prevMove[0] != "U":
                    speed += overwork(rRing, "dflick")
                    speed += overwork(rMiddle, "home")
                    speed += ringMult * pushMult
                    rRing = [speed, "home"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "B'":
                if lWrist == 1:
                    speed += overwork(lIndex, "home")
                    speed += 1
                    lIndex = [speed, "uflick"]
                elif rWrist == -1:
                    speed += overwork(rRing, "home")
                    speed += overwork(rMiddle, "home")
                    if prevMove[0] == "U":
                        speed += moveblock * 0.5 + ringMult
                    else:
                        speed += ringMult
                    rRing = [speed, "dflick"]
                elif rWrist == 1 and prevMove[0] not in ["U", "D"]:
                    if rIndex[1] == "uflick":
                        speed += overwork(rIndex, "eido", 0.75 * overWorkMult)
                        speed = max(speed, rOhCool + 2.5)
                    else:
                        speed += overwork(rIndex, "eido", 1.25 * overWorkMult)
                    speed += 1.15 * pushMult
                    rIndex = [speed, "uflick"]
                    rOhCool = speed
                elif rWrist == 0 and (lWrist == 1 or lWrist == -1):
                    speed += overwork(rIndex, "top", 0.9 * overWorkMult)
                    if prevMove[0] == "U":
                        speed += 1.45
                    else:
                        speed += 1
                    rIndex = [speed, "rightdb"]
                elif lWrist == -1 and prevMove[0] != "U":
                    speed += overwork(lRing, "dflick")
                    speed += overwork(lMiddle, "home")
                    speed += ringMult * pushMult
                    lRing = [speed, "home"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "B2":
                if rWrist == 1 and (lWrist != 1 or
                    max(overwork(rIndex, "home"), overwork(rMiddle, "home"), overwork(rRing, "u2grip", moveblock*overWorkMult)) <=
                    max(overwork(lIndex, "home"), overwork(lMiddle, "home"), overwork(lRing, "u2grip", moveblock*overWorkMult))):
                    speed += overwork(rIndex, "home")
                    speed += overwork(rMiddle, "home")
                    speed += overwork(rRing, "u2grip", moveblock * overWorkMult)
                    speed += double
                    rIndex = [speed, "uflick"]
                    rMiddle = [speed, "uflick"]
                elif lWrist == 1:
                    speed += overwork(lIndex, "home")
                    speed += overwork(lMiddle, "home")
                    speed += overwork(lRing, "u2grip", moveblock * overWorkMult)
                    speed += double
                    lIndex = [speed, "uflick"]
                    lMiddle = [speed, "uflick"]
                elif lWrist == -1 and (rWrist != -1 or
                    max(overwork(rMiddle, "home"), overwork(rRing, "home")) >
                    max(overwork(lMiddle, "home"), overwork(lRing, "home"))):
                    speed += overwork(lMiddle, "home")
                    speed += overwork(lRing, "home")
                    if prevMove[0] == "U":
                        speed += moveblock * 0.5 + double * ringMult
                    else:
                        speed += double * ringMult
                    lRing = [speed, "dflick"]
                elif rWrist == -1:
                    speed += overwork(rMiddle, "home")
                    speed += overwork(rRing, "home")
                    if prevMove[0] == "U":
                        speed += moveblock * 0.5 + double * ringMult
                    else:
                        speed += double * ringMult
                    rRing = [speed, "dflick"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "S":
                if rWrist == 0 and (lWrist != 0 or
                    overwork(rIndex, "top", 1.25 * overWorkMult) <= (moveblock * 0.5 + pushMult - 1) * sesliceMult):
                    speed += overwork(rIndex, "top", 1.25 * overWorkMult)
                    speed += sesliceMult
                    rIndex = [speed, "sflick"]
                elif lWrist == 0 and rWrist == -1:
                    speed += overwork(rIndex, "home", 1.25 * overWorkMult)
                    speed += overwork(rThumb, "top", 1.25 * overWorkMult)
                    speed += overwork(rMiddle, "home", 1.25 * overWorkMult)
                    speed += sesliceMult
                    rThumb = [speed, "top"]
                    rMiddle = [speed, "eflick"]
                elif lWrist == 0 and (rWrist == 0 or (rWrist == 1 and (prevMove in ["R", "L"]))):
                    speed += overwork(lIndex, "uflick", 1.25 * overWorkMult)
                    if prevMove[0] == "U":
                        speed += moveblock * 0.5 + pushMult * sesliceMult
                    else:
                        speed += pushMult * sesliceMult
                    lIndex = [speed, "top"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "S'":
                if lWrist == 0 and (rWrist != 0 or
                    overwork(lIndex, "top", 1.25 * overWorkMult) <= (moveblock * 0.5 + pushMult - 1) * sesliceMult):
                    speed += overwork(lIndex, "top", 1.25 * overWorkMult)
                    speed += sesliceMult
                    lIndex = [speed, "sflick"]
                elif rWrist == 0 and lWrist == -1:
                    speed += overwork(lIndex, "home", 1.25 * overWorkMult)
                    speed += overwork(lThumb, "bottom", 1.25 * overWorkMult)
                    speed += overwork(lMiddle, "home", 1.25 * overWorkMult)
                    speed += sesliceMult
                    lThumb = [speed, "top"]
                    lMiddle = [speed, "eflick"]
                elif rWrist == 0 and (lWrist == 0 or (lWrist == 1 and (prevMove in ["R", "L"]))):
                    speed += overwork(rIndex, "uflick", 1.25 * overWorkMult)
                    if prevMove[0] == "U":
                        speed += moveblock * 0.5 + pushMult * sesliceMult
                    else:
                        speed += pushMult * sesliceMult
                    rIndex = [speed, "top"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "S2":
                if (rWrist == -1 or rWrist == 1) and lWrist == 0:
                    speed += overwork(rThumb, "home")
                    speed += overwork(rIndex, "home")
                    speed += overwork(rMiddle, "home")
                    speed += overwork(rRing, "u2grip", moveblock * overWorkMult)
                    speed += sesliceMult * double
                    rMiddle = [speed, "e"]
                    rIndex = [speed, "e"]
                elif (lWrist == -1 or lWrist == 1) and rWrist == 0:
                    speed += overwork(lThumb, "home")
                    speed += overwork(lIndex, "home")
                    speed += overwork(lMiddle, "home")
                    speed += overwork(lRing, "u2grip", moveblock * overWorkMult)
                    speed += sesliceMult * double
                    rMiddle = [speed, "e"]
                    rIndex = [speed, "e"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "E":
                if (rWrist == 1 or rWrist == -1) and lWrist == 0:
                    speed += overwork(lIndex, "home")
                    speed += sesliceMult
                    lIndex = [speed, "e"]
                elif (lWrist == 1 or lWrist == -1) and rWrist == 0 and prevMove[0] != "B":
                    speed += overwork(rIndex, "e")
                    speed += sesliceMult * pushMult
                    rIndex = [speed, "home"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "E'":
                if (lWrist == 1 or lWrist == -1) and rWrist == 0:
                    speed += overwork(rIndex, "home")
                    speed += sesliceMult
                    rIndex = [speed, "e"]
                elif (rWrist == 1 or rWrist == -1) and lWrist == 0 and prevMove[0] != "B":
                    speed += overwork(lIndex, "e")
                    speed += sesliceMult * pushMult
                    lIndex = [speed, "home"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "E2":
                if (lWrist == 1 or lWrist == -1) and rWrist == 0:
                    speed += overwork(rIndex, "home")
                    speed += overwork(rMiddle, "home")
                    speed += overwork(rRing, "u2grip", moveblock * overWorkMult)
                    speed += sesliceMult * double
                    rIndex = [speed, "e"]
                    rMiddle = [speed, "e"]
                elif (rWrist == 1 or rWrist == -1) and lWrist == 0:
                    speed += overwork(lIndex, "home")
                    speed += overwork(lMiddle, "home")
                    speed += overwork(lRing, "u2grip", moveblock * overWorkMult)
                    speed += sesliceMult * double
                    lIndex = [speed, "e"]
                    lMiddle = [speed, "e"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "M'":
                if lWrist == 0:
                    speed += overwork(lThumb, "home")
                    speed += overwork(lIndex, "m")
                    speed += overwork(lMiddle, "m")
                    speed += overwork(lRing, "m")
                    if prevMove[0] == "B":
                        speed += 1.8
                    else:
                        speed += 1
                    lThumb = [speed, "home"]
                    lIndex = [speed, "m"]
                    lMiddle = [speed, "mflick"]
                    lRing = [speed, "m"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "M":
                if lWrist == 0 and prevMove[0] != "B":
                    speed += overwork(lThumb, "home")
                    speed += overwork(lIndex, "m")
                    speed += overwork(lMiddle, "mflick", 1.25 * overWorkMult)
                    speed += overwork(lRing, "m")
                    speed += pushMult
                    lThumb = [speed, "home"]
                    lIndex = [speed, "m"]
                    lMiddle = [speed, "m"]
                    lRing = [speed, "m"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "M2":
                if lWrist == 0:
                    speed += overwork(lThumb, "home")
                    speed += overwork(lIndex, "m")
                    speed += overwork(lMiddle, "m")
                    speed += overwork(lRing, "m")
                    if prevMove[0] == "B":
                        speed += moveblock + double
                    else:
                        speed += double
                    lThumb = [speed, "home"]
                    lIndex = [speed, "m"]
                    lMiddle = [speed, "mflick"]
                    lRing = [speed, "m"]
                else:
                    return [j, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "X":
                lWrist += 1
                rWrist += 1
                if lWrist > 1 or rWrist > 1:
                    return [j+1, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "X'":
                lWrist -= 1
                rWrist -= 1
                if lWrist < -1 or rWrist < -1:
                    return [j+1, speed, lWrist, rWrist,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove == "X2":
                if lWrist >= 1 and rWrist >= 1:
                    lWrist -= 2
                    rWrist -= 2
                elif lWrist <= -1 and rWrist <= -1:
                    lWrist += 2
                    rWrist += 2
                elif lWrist + rWrist > 0:
                    return [j, speed, lWrist-2, rWrist-2,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]
                else:
                    return [j, speed, lWrist+2, rWrist+2,
                            max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                            max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove in ["Y", "Y'", "Z", "Z'"]:
                speed += rotation
                return [j+1, speed, 0, 0,
                        max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                        max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            elif normalMove in ["Y2", "Z2"]:
                speed += rotation * double
                return [j+1, speed, 0, 0,
                        max(lThumb[0], lIndex[0], lMiddle[0], lRing[0]),
                        max(rThumb[0], rIndex[0], rMiddle[0], rRing[0])]

            else:
                return "Unknown move: " + move

            # ----- Post-move adjustments (grip changes, speed corrections) -----
            if firstMoveSpeed is not None:
                speed = max(firstMoveSpeed, speed) + 0.5
                prevSpeed = None
                firstMoveSpeed = None

            if (move[0] == "R" or move[0] == "l") and grip == -1:
                grip = 1
                speed += 0.65
            elif (move[0] == "r" or move[0] == "L") and grip == 1:
                grip = -1
                speed += 0.65

            if move[0] == "d" and udgrip == -1:
                udgrip = 1
                speed += 2.25
            elif (move[0] == "U" or move[0] == "u") and udgrip == 1:
                udgrip = -1
                speed += 2.25

            if j >= 2:
                prev2 = splitSeq[j-2].upper()
                prev1 = splitSeq[j-1].upper()
                if (normalMove == "R" and move == splitSeq[j-2] and prev1 == "U'") or \
                   (normalMove == "R'" and move == splitSeq[j-2] and prev1 == "U"):
                    speed -= 0.5
                elif (normalMove == "R" and move == splitSeq[j-2] and prev1 == "D'" and rWrist == 1) or \
                     (normalMove == "R'" and move == splitSeq[j-2] and prev1 == "D"):
                    speed -= 0.3

            if normalMove == "U" and (lWrist == -1 or rWrist == -1):
                speed += destabilize
            if normalMove == "B" and (lWrist == 0 or rWrist == 0):
                speed += destabilize
            if normalMove == "D" and (lWrist == 1 or rWrist == 1):
                speed += destabilize
            if normalMove == "S" and (lWrist == 1 or rWrist == 1 or lWrist == -1 or rWrist == -1):
                speed += destabilize
            if normalMove == "E" and (lWrist == 0 or rWrist == 0):
                speed += destabilize

        return [-1, speed, lGrip, rGrip]

    # ----- Main function body (unchanged) -----
    splitSeq = sequence.split()
    trueSplitSeq = []
    for token in splitSeq:
        if ignoreErrors:
            if token.lower() in ["r","r2","r'","u","u'","u2","f","f2","f'","d","d2","d'","l","l2","l'","b","b2","b'","m","m2","m'","s","s2","s'","e","e2","e'","x","x'","x2","y","y'","y2","z","z'","z2"]:
                trueSplitSeq.append(token)
        else:
            if token != "":
                trueSplitSeq.append(token)
    splitSeq = trueSplitSeq[:]

    if ignoreauf:
        if splitSeq and splitSeq[0][0] == "U":
            splitSeq.pop(0)
        elif len(splitSeq) >= 2 and splitSeq[0][0].lower() == "d" and splitSeq[1][0] == "U":
            splitSeq[1] = splitSeq[0]
            splitSeq.pop(0)
        if splitSeq and splitSeq[-1][0] == "U":
            splitSeq.pop()
        elif len(splitSeq) >= 2 and splitSeq[-1][0].lower() == "d" and splitSeq[-2][0] == "U":
            splitSeq[-2] = splitSeq[-1]
            splitSeq.pop()

    tests = [
        test(splitSeq, 0, 0, 0),
        test(splitSeq, 0, -1, 1 + addRegrip),
        test(splitSeq, 0, 1, 1 + addRegrip),
        test(splitSeq, -1, 0, 1 + addRegrip),
        test(splitSeq, 1, 0, 1 + addRegrip)
    ]

    while True:
        for t in tests:
            if t[0] == "U":
                return t

        bestTest = tests[0]
        for compTest in tests[1:]:
            if compTest[0] == -1 and (bestTest[0] != -1 or bestTest[1] > compTest[1]):
                bestTest = compTest
            elif compTest[0] > bestTest[0] and bestTest[0] != -1:
                bestTest = compTest
            elif compTest[0] == bestTest[0] and compTest[1] < bestTest[1] and bestTest[0] != -1:
                bestTest = compTest

        if bestTest[0] == -1:
            return round(bestTest[1] * 10) / 10

        new_tests = []
        prevMoveType = splitSeq[bestTest[0] - 1][0] if bestTest[0] >= 1 else " "
        prev2Type = splitSeq[bestTest[0] - 2][0] if bestTest[0] >= 2 else " "
        doubleRegrip = (bestTest[2] > 1 or bestTest[2] < -1) and (bestTest[3] > 1 or bestTest[3] < -1)

        for leftWrist in range(-1, 2):
            for rightWrist in range(-1, 2):
                leftMatch = (bestTest[2] == leftWrist)
                rightMatch = (bestTest[3] == rightWrist)
                if prevMoveType in ["X","x","Y","y","Z","z"]:
                    new_tests.append(test(splitSeq[bestTest[0]:], leftWrist, rightWrist, bestTest[1]))
                else:
                    if doubleRegrip:
                        penalty = rotation * double
                    else:
                        penalty = 2
                    rMoveLatency = 1 if (prevMoveType in ["R","r"] or prev2Type in ["R","r"]) else 0
                    lMoveLatency = 1 if (prevMoveType in ["L","l"] or prev2Type in ["L","l"]) else 0
                    if leftMatch or doubleRegrip:
                        rHandLatency = max(0, 2 - (bestTest[1] - bestTest[5]))
                        penalty = max(rHandLatency, rMoveLatency, lMoveLatency * 2)
                        new_tests.append(test(splitSeq[bestTest[0]:], leftWrist, rightWrist,
                                              bestTest[1] + penalty + addRegrip))
                    elif rightMatch or doubleRegrip:
                        lHandLatency = max(0, 2 - (bestTest[1] - bestTest[4]))
                        penalty = max(lHandLatency, lMoveLatency, rMoveLatency * 2)
                        new_tests.append(test(splitSeq[bestTest[0]:], leftWrist, rightWrist,
                                              bestTest[1] + penalty + addRegrip))
        tests = new_tests
        splitSeq = splitSeq[bestTest[0]:]

# ------------------- Example usage -------------------
if __name__ == "__main__":
    # Quick test
    print(alg_speed("R D' R' D'"))