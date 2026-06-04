# -*- coding: utf-8 -*-
import os, ast

def check_py_syntax():
    errors = []
    for root, dirs, files in os.walk(chr(46)):
        dirs[:] = [d for d in dirs if d not in (chr(46)+chr(118)+chr(101)+chr(110)+chr(118), chr(95)+chr(95)+chr(112)+chr(121)+chr(99)+chr(104)+chr(97)+chr(99)+chr(104)+chr(101)+chr(95)+chr(95), chr(46)+chr(103)+chr(105)+chr(116))]
        for f in files:
            if not f.endswith(chr(46)+chr(112)+chr(121)): continue
            path = os.path.join(root, f)
            try:
                with open(path, chr(114), encoding=chr(117)+chr(116)+chr(102)+chr(45)+chr(56), errors=chr(114)+chr(101)+chr(112)+chr(108)+chr(97)+chr(99)+chr(101)) as fh:
                    ast.parse(fh.read())
            except SyntaxError as e:
                errors.append(path + chr(58) + chr(32) + str(e))
    return errors

def check_md_math():
    missing = []
    td = os.path.join(chr(46), chr(25945)+chr(23398)+chr(25991)+chr(26723))
    for root, dirs, files in os.walk(td):
        for f in files:
            if not f.endswith(chr(46)+chr(109)+chr(100)): continue
            path = os.path.join(root, f)
            try:
                with open(path, chr(114), encoding=chr(117)+chr(116)+chr(102)+chr(45)+chr(56)) as fh:
                    content = fh.read()
                marker = chr(25968)+chr(23398)+chr(21407)+chr(29702)
                if marker not in content:
                    missing.append(path)
            except Exception:
                missing.append(path + chr(32)+chr(114)+chr(101)+chr(97)+chr(100)+chr(32)+chr(101)+chr(114)+chr(114)+chr(111)+chr(114))
    return missing

def main():
    print(chr(61) * 60)
    print(chr(77)+chr(76)+chr(45)+chr(80)+chr(108)+chr(97)+chr(121)+chr(103)+chr(114)+chr(111)+chr(117)+chr(110)+chr(100)+chr(32)+chr(65)+chr(117)+chr(116)+chr(111)+chr(109)+chr(97)+chr(116)+chr(101)+chr(100)+chr(32)+chr(67)+chr(104)+chr(101)+chr(99)+chr(107))
    print(chr(61) * 60)
    print()
    print(chr(49)+chr(46)+chr(32)+chr(80)+chr(121)+chr(116)+chr(104)+chr(111)+chr(110)+chr(32)+chr(115)+chr(121)+chr(110)+chr(116)+chr(97)+chr(120)+chr(32)+chr(99)+chr(104)+chr(101)+chr(99)+chr(107))
    py_errors = check_py_syntax()
    if py_errors:
        for e in py_errors:
            print(chr(32)*4 + chr(91)+chr(69)+chr(82)+chr(82)+chr(79)+chr(82)+chr(93) + chr(32) + e)
    else:
        print(chr(32)*4 + chr(65)+chr(108)+chr(108)+chr(32)+chr(46)+chr(112)+chr(121)+chr(32)+chr(102)+chr(105)+chr(108)+chr(101)+chr(115)+chr(32)+chr(112)+chr(97)+chr(115)+chr(115)+chr(32)+chr(115)+chr(121)+chr(110)+chr(116)+chr(97)+chr(120)+chr(32)+chr(99)+chr(104)+chr(101)+chr(99)+chr(107)+chr(33))
    print()
    print(chr(50)+chr(46)+chr(32)+chr(84)+chr(101)+chr(97)+chr(99)+chr(104)+chr(105)+chr(110)+chr(103)+chr(32)+chr(100)+chr(111)+chr(99)+chr(115)+chr(32)+chr(45)+chr(32)+chr(109)+chr(97)+chr(116)+chr(104)+chr(32)+chr(115)+chr(101)+chr(99)+chr(116)+chr(105)+chr(111)+chr(110)+chr(32)+chr(99)+chr(104)+chr(101)+chr(99)+chr(107))
    md_missing = check_md_math()
    if md_missing:
        for m in md_missing:
            print(chr(32)*4 + chr(91)+chr(77)+chr(73)+chr(83)+chr(83)+chr(73)+chr(78)+chr(71)+chr(93) + chr(32) + m)
    else:
        print(chr(32)*4 + chr(65)+chr(108)+chr(108)+chr(32)+chr(46)+chr(109)+chr(100)+chr(32)+chr(102)+chr(105)+chr(108)+chr(101)+chr(115)+chr(32)+chr(104)+chr(97)+chr(118)+chr(101)+chr(32)+chr(109)+chr(97)+chr(116)+chr(104)+chr(32)+chr(115)+chr(101)+chr(99)+chr(116)+chr(105)+chr(111)+chr(110)+chr(33))
    print()
    total = len(py_errors) + len(md_missing)
    print(chr(61) * 60)
    if total == 0:
        print(chr(65)+chr(108)+chr(108)+chr(32)+chr(99)+chr(104)+chr(101)+chr(99)+chr(107)+chr(115)+chr(32)+chr(112)+chr(97)+chr(115)+chr(115)+chr(101)+chr(100)+chr(33))
    else:
        print(chr(70)+chr(111)+chr(117)+chr(110)+chr(100)+chr(32) + str(total) + chr(32)+chr(105)+chr(115)+chr(115)+chr(117)+chr(101)+chr(115)+chr(46))
    return total

if __name__ == chr(95)+chr(95)+chr(109)+chr(97)+chr(105)+chr(110)+chr(95)+chr(95):
    import sys
    sys.exit(main())