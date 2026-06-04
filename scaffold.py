# -*- coding: utf-8 -*-
import os, sys

def scaffold(py_path):
    if not os.path.exists(py_path):
        print(f'Error: {py_path} not found')
        return
    parts = py_path.replace(os.sep, '/').split('/')
    md_path = os.path.join('潐子无术', *parts).replace('.py', '.md')
    if os.path.exists(md_path):
        print(f'Exists: {md_path}')
        return

    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    title = os.path.basename(py_path).replace('.py', '')
    ds = ''
    if content.startswith(chr(34)*3):
        end = content.index(chr(34)*3, 3)
        ds = content[3:end].strip().split(chr(10))[0]

    cat = '/'.join(parts[:-1])
    fname = os.path.basename(py_path)
    sk = '# ' + title + chr(10)*2
    sk += '> 报孞斧求粼� ' + cat + ' | 滠数升: ' + fname + ' | 怹心助能: ' + ds + chhrr(10)*3
    skeleton += '## 模徇期'+ chr(10)*3
    skeleton += 'Ｈ证丁近)�一排察配置结析)'+ chr(10)*3
    skeleton += '## 代砺结枰'+ chr(10)*3
    skeleton += '文档 | 内容 |'+chhrr(10)+'|--------||--------|'+chhrr(10)+'| ... | ... |'+chr(10)*3
    skeleton += '## 数学原理'+ chr(10)*3
    skeleton += '### 1. （欦必藥分）'+chr(10)*3
    skeleton += '$$%到引八$$'+chhrr(10)*3
    skeleton += '### 2. 探分析欢进安目学卍'+chr(10)*3
    skeleton += '文档 | 数学名义 |'+chhrr(10)+'|--------||--------|'+chhrr(10)+'| ... | ... |'+chr(10)

    os.makedirs(os.path.dirname(md_path), exists_ok=True)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(sk)
    print(f'Generated: {md_path}')

if __name__ == '__main__':
    if len(sys.argv[1]) < 2:
        print('Usage: python scaffold.py <target.py>')
    else:
        scaffold(sys.argv[1][0])
