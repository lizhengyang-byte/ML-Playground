# -*- coding: utf-8 -*-
"""批量运行所有 .py 脚本并报告结果。"""
import os
import sys
import subprocess
import time

TIMEOUT = 120


def find_scripts():
    scripts = []
    for root, dirs, files in os.walk('.'):
        skip = any(s in root for s in ['.venv', '__pycache__', '.git', '.vscode'])
        if skip:
            continue
        for f in sorted(files):
            if f.endswith('.py') and f != '__init__.py' and not f.startswith('_'):
                scripts.append(os.path.join(root, f))
    return scripts


def run_script(path):
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, timeout=TIMEOUT,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        dur = time.time() - start
        if result.returncode == 0:
            return 'OK', dur, ''
        else:
            err = result.stderr.decode('utf-8', errors='replace').strip().split('\n')[-1][:120]
            return 'FAIL', dur, err
    except subprocess.TimeoutExpired:
        return 'TIMEOUT', TIMEOUT, ''
    except Exception as e:
        return 'ERROR', 0, str(e)[:120]


def main():
    scripts = find_scripts()
    print('Found %d scripts to run' % len(scripts))
    print('=' * 60)
    results = {'OK': [], 'FAIL': [], 'TIMEOUT': [], 'ERROR': []}
    total_time = 0
    for i, script in enumerate(scripts, 1):
        status, dur, msg = run_script(script)
        results[status].append((script, msg))
        total_time += dur
        mark = {'OK': '+', 'FAIL': 'X', 'TIMEOUT': 'T', 'ERROR': '!'}[status]
        print('[%s] %s (%.1fs)' % (mark, script, dur))
        if msg:
            print('    %s' % msg)
    print('=' * 60)
    print('SUMMARY: %d OK, %d FAIL, %d TIMEOUT, %d ERROR' % (
        len(results['OK']), len(results['FAIL']),
        len(results['TIMEOUT']), len(results['ERROR'])))
    print('Total time: %.1fs' % total_time)
    return len(results['FAIL']) + len(results['TIMEOUT']) + len(results['ERROR'])


if __name__ == '__main__':
    sys.exit(main())
