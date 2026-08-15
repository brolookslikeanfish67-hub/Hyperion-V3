"""
Hyperion-V3: Tool Inventory Suite.
Provides OS execution sandboxes.
"""
import os
import sys
import io
import traceback
import subprocess

class HyperionTools:
    @staticmethod
    def execute_python_code(
        code: str
    ) -> str:
        """Executes code and gets stdout/stderr."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_out = io.StringIO()
        redirected_err = io.StringIO()
        
        sys.stdout = redirected_out
        sys.stderr = redirected_err
        
        try:
            exec(code, {"__builtins__": __builtins__})
            result = redirected_out.getvalue()
        except Exception:
            result = traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
        return result if result.strip() else "Success (No Output)"

    @staticmethod
    def read_local_file(
        path: str
    ) -> str:
        """Reads code files for context parsing."""
        if not os.path.exists(path):
            return f"Error: {path} not found."
        try:
            with open(
                path, "r", encoding="utf-8"
            ) as f:
                return f.read()
        except Exception as e:
            return str(e)

    @staticmethod
    def write_local_file(
        path: str, content: str
    ) -> str:
        """Writes/patches code file matrix lines."""
        try:
            os.makedirs(
                os.path.dirname(path), 
                exist_ok=True
            )
            with open(
                path, "w", encoding="utf-8"
            ) as f:
                f.write(content)
            return f"Successfully written to {path}"
        except Exception as e:
            return str(e)

    @staticmethod
    def run_terminal_command(
        command: str
    ) -> str:
        """Runs terminal linters or compilers."""
        try:
            res = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=10
            )
            return f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        except Exception as e:
            return str(e)
