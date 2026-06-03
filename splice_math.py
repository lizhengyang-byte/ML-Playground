import sys
def replace_math(target_path, new_math_path):
    with open(target_path, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(new_math_path, 'r', encoding='utf-8') as f:
        new_math = f.read()
    marker = '## 数学原理'
    idx = content.find(marker)
    if idx == -1:
        print(f'WARNING: no math section in {target_path}')
        return
    new_content = content[:idx] + new_math
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f'OK: {target_path}')

if __name__ == '__main__':
    replace_math(sys.argv[1], sys.argv[2])
