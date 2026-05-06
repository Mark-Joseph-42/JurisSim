import subprocess
import tempfile
import os

class LogicSolver:
    def verify_code(self, z3_python_code: str) -> dict:
        """
        Executes the given Python-Z3 code in a temporary file and captures the output.
        Returns a dict with 'stdout', 'stderr', and 'result' (sat/unsat/error).
        """
        # Basic input sanitization
        forbidden = ["import os", "import subprocess", "import sys", "__import__", "eval(", "exec(", "open("]
        for term in forbidden:
            if term in z3_python_code:
                return {"result": "error", "stdout": "", "stderr": f"Security violation: {term} is forbidden."}

        # Ensure the code imports z3
        if "from z3 import *" not in z3_python_code and "import z3" not in z3_python_code:
            z3_python_code = "from z3 import *\n" + z3_python_code

        # Write code to a temporary file
        fd, path = tempfile.mkstemp(suffix=".py")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(z3_python_code)
            
            # Execute the file
            process = subprocess.run(
                ["python", path],
                capture_output=True,
                text=True,
                timeout=10 # Prevent infinite loops
            )
            
            stdout = process.stdout.strip()
            stderr = process.stderr.strip()
            
            if process.returncode != 0:
                return {"result": "error", "stdout": stdout, "stderr": stderr}
            
            # Parse sat/unsat from stdout
            result = "unknown"
            if "sat" in stdout and "unsat" not in stdout:
                result = "sat"
            elif "unsat" in stdout:
                result = "unsat"
                
            return {"result": result, "stdout": stdout, "stderr": stderr}
            
        except subprocess.TimeoutExpired:
            return {"result": "error", "stdout": "", "stderr": "Execution timed out."}
        finally:
            os.remove(path)
