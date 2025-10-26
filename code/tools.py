from RestrictedPython import compile_restricted, safe_globals

from langchain.tools import tool


def safe_execute(code):
    byte_code = compile_restricted(code, '<string>', 'exec')
    env = safe_globals.copy()
    env['_print_'] = print
    env['result'] = None

    exec(byte_code, env)
    return str(env.get('result', 'No result variable defined.'))



@tool("execute_python_code", return_direct=True)
def execute_python_code(code: str) -> str:
    """
    Safely execute a restricted subset of Python code.
    The code should define a variable named `result` to return.
    Printing is not allowed.
    """
    return safe_execute(code)