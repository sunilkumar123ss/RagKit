#!/usr/bin/env python3
"""Verify this repository's tests, synthetic demo and command-line help."""
from pathlib import Path
import json, re, subprocess, sys
ROOT=Path(__file__).resolve().parent
config=json.loads((ROOT/'project.json').read_text())
checks=[]
for name, args, expected in [
    ('tests',['-m','unittest','discover','-s','.','-p','test_main.py','-v'],0),
    ('demo',['main.py']+config['demo_args'],config['expected_exit']),
    ('help',['main.py','--help'],0)]:
    result=subprocess.run([sys.executable]+args,cwd=ROOT,text=True,capture_output=True,timeout=30)
    count=re.search(r'Ran (\d+) tests?',result.stderr) if name=='tests' else None
    passed=result.returncode==expected and (name!='tests' or count and int(count.group(1))>0)
    checks.append(bool(passed))
    print(('PASS' if passed else 'FAIL')+' '+name)
    if not passed: print(result.stdout+result.stderr)
sys.exit(0 if all(checks) else 1)
