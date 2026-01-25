import sys
import pathlib
from pathlib import Path
import re
from typing import Any, Dict

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

sys.path.insert(0, str(pathlib.Path().resolve()))
from generator.env import InteractiveContainerEnvironment
from generator.sample_solutions import _extract_action
import time


class SkyRLContainerEnv(BaseTextEnv):
    """Graph navigation environment compatible with SkyRL."""

    def __init__(self, env_config: Dict[str, Any] | None = None, extras: Dict[str, Any] | None = None):
        super().__init__()
        env_config = env_config
        task_dir = extras["extra_info"].get("task_dir")
        task_dir = Path(task_dir)

        container_sif_path = task_dir / "container.sif"
        initial_test_path = task_dir / "test_initial_state.py"
        final_test_path = task_dir / "test_final_state.py"
        def_path = task_dir / "container.def"
        self.start_time = time.time()
        self.max_time = extras["extra_info"].get("max_time", 600)
        # check if max_time is a string and convert to int
        if isinstance(self.max_time, str):
            self.max_time = int(self.max_time)
        self.reward = 0

        self.max_turns = extras.get("max_turns", 16)
        # Make verbose mode configurable (default to False for better performance)
        verbose_mode = extras["extra_info"].get("verbose", False)
        # Output truncation limit (default 50K chars to prevent memory issues)
        self.max_output_length = extras["extra_info"].get("max_output_length", 50000)
        
        self.env = InteractiveContainerEnvironment(
            container_sif_path=container_sif_path,
            initial_test_path=initial_test_path,
            final_test_path=final_test_path,
            def_path=def_path,
            verbose=verbose_mode,
        )
        # Lazy initialization: don't initialize in __init__ to prevent delayed Ray actor 
        # creations from spawning containers during training phases
        self._initialized = False

    def __del__(self):
        """Cleanup on destruction if environment was initialized."""
        if hasattr(self, '_initialized') and self._initialized:
            try:
                self.env.cleanup()
            except Exception:
                pass  # Best effort cleanup

    def step(self, action: str) -> BaseTextEnvStepOutput:
        # Lazy initialization: only initialize when first step is called
        # Also check if environment was cleaned up (instance_name would be None)
        if not self._initialized or self.env.instance_name is None:
            # If environment was cleaned up, we can't reuse it - return error
            if self.env.instance_name is None and self._initialized:
                return BaseTextEnvStepOutput(
                    observations=[{"role": "user", "content": "❌ Environment was cleaned up and cannot be reused"}],
                    reward=0.0,
                    done=True,
                    metadata={"goal_reached": False, "env_cleaned_up": True},
                )
            init_success = self.env.initialize(run_initial_tests=False)
            if not init_success:
                # If initialization fails, return error and mark as done
                return BaseTextEnvStepOutput(
                    observations=[{"role": "user", "content": "❌ Failed to initialize container environment"}],
                    reward=0.0,
                    done=True,
                    metadata={"goal_reached": False, "init_failed": True},
                )
            self._initialized = True
        
        self.turns += 1
        action = _extract_action(action)
        goal_reached = False

        done = False
        if action["type"] == "done":
            done = True
            result_back = "Done"

        elif action["type"] == "command":
            command = action["command"] or ""
            success, output = self.env.exec(command)
            
            # Truncate very long outputs to prevent memory issues
            truncated_msg = ""
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length]
                truncated_msg = f"\n[Output truncated: showing first {self.max_output_length} of {len(output)} characters]"
            
            if success:
                result_back = f"Command executed successfully. Output: {output}{truncated_msg}\n\n(exit_code={0 if success else 1})"
            else:
                result_back = f"Command failed. Output: {output}{truncated_msg}\n\n(exit_code={0 if success else 1})"

        else:
            result_back = "Could not parse a single <command>...</command> or <action>done</action>. Please respond with exactly one of those."
        # Check termination conditions
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        timed_out = False
        
        if self.turns >= self.max_turns:
            done = True
        elif elapsed_time > self.max_time:
            done = True
            timed_out = True

        if self.env.verbose:
            print(f"Time taken so far: {elapsed_time:.2f}s")
        
        # Handle done state
        if done:
            # Only run tests and cleanup if environment was actually initialized
            if self._initialized:
                if not timed_out:
                    # Run tests only if not timed out
                    success, test_output = self.env.run_final_tests()
                    goal_reached = success
                    if success:
                        self.reward += 1
                # Always cleanup when done
                self.env.cleanup()
                # Mark as not initialized so we don't try to reuse this environment
                self._initialized = False

        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": result_back}],
            reward=self.reward,
            done=done,
            metadata={"goal_reached": goal_reached},
        )
