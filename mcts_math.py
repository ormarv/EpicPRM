import re
import os
import tempfile
from typing import List, Dict, Any, Optional, Type, Tuple, Union
import psutil
from subprocess import TimeoutExpired, Popen, PIPE

def extract_code_blocks(text):
    import re
    pattern = r'```(\w+)\n([\s\S]*?)```'
    matches = re.finditer(pattern, text)
    results = []
    response  = ''
    
    for match in matches:
        code = match.group(2)
        response+=code.strip()+'\n'    
    return response


    
def test_pass(test_case, func_str):
    func_str = extract_code_blocks(func_str)
    
    func_name = func_str.split('def')[1].split('(')[0].strip()
    refined_test = []
    for line in test_case.splitlines():
        if line.startswith('from tmp import') or line.startswith('from your_module import'):
            continue
        if 'import' in line and func_name in line:
            continue
        refined_test.append(line)
    extracted_test = '\n'.join(refined_test)

    code = func_str + "\n" + extracted_test
    
    def kill(proc_pid:int):
        process= psutil.Process(proc_pid)
        
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()
    
    def pexec(strcmd:str,n_timeout:int):
        try:
            p=Popen(strcmd,shell=True, stdout=PIPE, stderr=PIPE)
            stdout, stderr = p.communicate(timeout=n_timeout)
            return stdout.decode('utf-8'), stderr.decode('utf-8'), p.returncode
        except TimeoutExpired:
            kill(p.pid)
            return None, None,-1
        except Exception as e:
            return None, None,-1
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, dir='/tmp') as f:
        f.write(code.encode('utf-8'))
        temp_path = f.name

    temp_dir = os.path.dirname(temp_path)
    cmd = f'cd {temp_dir} ; python {temp_path}'
    stdout, stderr, returncode = pexec(cmd, 3)
    
    os.unlink(temp_path)
    
    test_pass = returncode == 0 and 'AssertionError' not in stderr
    return test_pass



def math_is_equiv(grt: Union[str, list[str]], prd: str):
    # return True
    if isinstance(grt, list):
        for g in grt:
            if not test_pass(g, prd):
                return False
        return True
    else:
        return test_pass(grt, prd)
