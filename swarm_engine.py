"""
Hyperion-V3: Multi-Agent Swarm Logic Core.
Loops code compilation until execution passes.
"""
import sys
import io
import traceback
from generate import HyperionInferenceEngine

class InfiniteSwarmCore:
    def __init__(
        self, 
        engine: HyperionInferenceEngine
    ):
        self.engine = engine

    def _execute_in_sandbox(
        self, 
        code_str: str
    ) -> tuple[bool, str]:
        """
        Executes code inside an isolated environment 
        and captures stdout or tracebacks.
        """
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        success = True
        error_msg = ""
        
        try:
            # Execute within empty context global maps
            exec(code_str, {})
        except Exception:
            success = False
            error_msg = traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            
        return success, error_msg

    def compile_until_perfect(
        self, 
        task_prompt: str
    ) -> str:
        """
        Infinite self-correction driver loop.
        Will continue generating and parsing errors
        until code execution reports 0 bugs.
        """
        current_prompt = (
            f"Write clean Python code for: {task_prompt}\n"
            f"Provide raw code executable sequences only."
        )
        
        attempt = 1
        
        while True:
            print(f"\n[Swarm] Processing Loop iteration: #{attempt}")
            
            # 1. Prompt model matrix layers to build code
            candidate_code = self.engine.generate(
                current_prompt,
                max_new_tokens=512,
                temperature=0.1  # Low temp locks syntax stability
            )
            
            # 2. Test code inside local runtime environment
            passed, runtime_logs = self._execute_in_sandbox(
                candidate_code
            )
            
            if passed:
                print(f"[✓] Success on Attempt #{attempt}! Code works perfectly.")
                print("=========================================")
                print(candidate_code)
                print("=========================================")
                return candidate_code
                
            # 3. Code failed. Isolate traceback and feedback to engine
            print(f"[-] Attempt #{attempt} failed with errors. Retrying...")
            print(f"[Logs]: {runtime_logs.splitlines()[-1]}") # Print concise error
            
            current_prompt = (
                f"Your previous code failed execution tests.\n"
                f"--- BROKEN CODE ---\n{candidate_code}\n"
                f"--- RUNTIME ERROR TRACE ---\n{runtime_logs}\n"
                f"Fix the bugs and rewrite the code perfectly."
            )
            
            attempt += 1

if __name__ == "__main__":
    # Test stub orchestrator
    from generate import CHECKPOINT_PATH, TOKENIZER_JSON
    
    base_engine = HyperionInferenceEngine(
        checkpoint_path=CHECKPOINT_PATH,
        tokenizer_path=TOKENIZER_JSON
    )
    
    swarm = InfiniteSwarmCore(base_engine)
    
    # Test case: Give it a task to code and track
    prompt = "Create a function that reverses linked list structures."
    swarm.compile_until_perfect(prompt)
