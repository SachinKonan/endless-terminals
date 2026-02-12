"""
Convert SIF files in task directories to Dockerfiles using LLM, build Docker images,
and verify that initial tests pass.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict, Any

from tqdm import tqdm
from openai import OpenAI

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# Initialize clients lazily
_openai_client = None
_anthropic_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is not installed. Run: pip install anthropic")
        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client

SYSTEM_MSG = """You are an expert in converting Apptainer or Singularity definition files to Dockerfiles.
Convert the given Apptainer .def file to an equivalent Dockerfile.

Key conversion rules:
- Apptainer's `Bootstrap: docker` with `From: ubuntu:22.04` becomes `FROM ubuntu:22.04`
- `%post` section commands become `RUN` commands in Dockerfile
- `%environment` section becomes `ENV` commands
- `%help` section can be added as comments
- Ensure all commands are properly formatted for Docker
- Use `RUN` for each command or chain them with `&&`
- Make sure to install pytest if it's needed
- Preserve user creation and permissions setup
- The home path should be /home/user
- Set WORKDIR to /home/user so commands run from there by default

Heredoc and multiline command rules (IMPORTANT):
- Every shell command must be inside a `RUN` instruction. Do NOT leave lines like `PermitRootLogin yes` or similar outside of a `RUN`; otherwise Docker will treat them as invalid instructions.
- When converting heredoc blocks (e.g. `cat <<EOF ... EOF`):
  - Keep the entire heredoc inside a single `RUN` instruction.
  - The line with `<<EOF` (or `<< 'EOF'`) must be the last thing on that line (no trailing `&&` etc.).
  - The heredoc body must start on the next line and be left as plain text.
  - The terminating `EOF` must be alone on its own line with nothing after it.
  - If needed, you may place any follow-up commands (e.g. `chmod ...`) in a separate `RUN` instruction after the heredoc.
- Do NOT wrap a heredoc block inside a single-quoted string like `sh -c 'cat <<EOF ... EOF'` in a way that causes the heredoc contents to be parsed as Dockerfile instructions.
- When adding comments, use the `#` character and place it at the beginning of the line.
- Always use ubuntu:22.04 from docker. Do not mention .sif containers.

Output only the Dockerfile content, no explanations or markdown code blocks."""


USER_TEMPLATE = """Convert this Apptainer definition file to a Dockerfile:

{def_content}

Output the complete Dockerfile only."""


def read_def_file(def_path: Path) -> Optional[str]:
    """Read the Apptainer definition file."""
    return def_path.read_text(encoding="utf-8")


def _extract_dockerfile_content(response_content: str) -> str:
    """Extract Dockerfile content from LLM response, removing markdown code blocks if present."""
    dockerfile_content = response_content.strip()
    
    # Remove markdown code block markers if present
    if "```dockerfile" in dockerfile_content or "```Dockerfile" in dockerfile_content:
        # Extract content between ```dockerfile and ```
        parts = dockerfile_content.split("```")
        for i, part in enumerate(parts):
            if "dockerfile" in part.lower() and i + 1 < len(parts):
                dockerfile_content = parts[i + 1].strip()
                break
    elif dockerfile_content.startswith("```"):
        # Generic code block
        lines = dockerfile_content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        dockerfile_content = "\n".join(lines)
    
    return dockerfile_content.strip()


def convert_def_to_dockerfile_openai(
    def_content: str,
    model: str = "gpt-5.1",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    max_retries: int = 5,
    base_delay: float = 10.0,
) -> Optional[str]:
    """Use OpenAI LLM to convert Apptainer definition to Dockerfile."""
    prompt = USER_TEMPLATE.format(def_content=def_content)
    client = get_openai_client()
    
    for attempt in range(max_retries):
        try:
            # Make API call
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            return _extract_dockerfile_content(response_content)
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "rate" in error_str or "429" in error_str or "overloaded" in error_str
            
            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"Rate limited (OpenAI), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                print(f"Error converting to Dockerfile with OpenAI: {e}")
                return None
    
    return None


def convert_def_to_dockerfile_anthropic(
    def_content: str,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    max_retries: int = 5,
    base_delay: float = 10.0,
) -> Optional[str]:
    """Use Anthropic LLM to convert Apptainer definition to Dockerfile."""
    prompt = USER_TEMPLATE.format(def_content=def_content)
    client = get_anthropic_client()
    
    for attempt in range(max_retries):
        try:
            # Make API call
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=SYSTEM_MSG,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            
            # Extract response content
            response_content = response.content[0].text
            return _extract_dockerfile_content(response_content)
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "rate" in error_str or "429" in error_str or "overloaded" in error_str
            
            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"Rate limited (Anthropic), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                print(f"Error converting to Dockerfile with Anthropic: {e}")
                return None
    
    return None


def convert_def_to_dockerfile(
    def_content: str,
    model: str = "gpt-5.1",
    provider: str = "openai",
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> Optional[str]:
    """Use LLM to convert Apptainer definition to Dockerfile.
    
    Args:
        def_content: The Apptainer definition file content.
        model: Model name to use (e.g., "gpt-5.1", "claude-sonnet-4-20250514", "claude-opus-4-20250514").
        provider: Either "openai" or "anthropic".
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
    
    Returns:
        Dockerfile content or None if conversion failed.
    """
    if provider == "anthropic":
        return convert_def_to_dockerfile_anthropic(
            def_content, model=model, temperature=temperature, max_tokens=max_tokens
        )
    else:
        return convert_def_to_dockerfile_openai(
            def_content, model=model, temperature=temperature, max_tokens=max_tokens
        )


def pre_pull_base_images(dockerfiles: List[Path]) -> None:
    """Pre-pull common base images to avoid redundant pulls during parallel builds."""
    base_images = set()
    for dockerfile_path in dockerfiles:
        if dockerfile_path.exists():
            content = dockerfile_path.read_text(encoding="utf-8")
            for line in content.split("\n"):
                line = line.strip()
                if line.upper().startswith("FROM "):
                    # Extract image name (handle "FROM image AS stage" syntax)
                    parts = line.split()
                    if len(parts) >= 2:
                        image = parts[1]
                        base_images.add(image)
    
    if base_images:
        print(f"Pre-pulling {len(base_images)} base images: {base_images}")
        for image in base_images:
            try:
                subprocess.run(
                    ["docker", "pull", image],
                    timeout=300,
                    capture_output=True,
                )
                print(f"  ✓ Pulled {image}")
            except Exception as e:
                print(f"  ⚠ Failed to pull {image}: {e}")


def create_dockerignore(context_dir: Path) -> None:
    """Create a .dockerignore to minimize build context transfer."""
    dockerignore_path = context_dir / ".dockerignore"
    if not dockerignore_path.exists():
        dockerignore_path.write_text(
            "*.sif\n"
            "*.tar\n"
            "*.tar.gz\n"
            "*.zip\n"
            "solutions/\n"
            "__pycache__/\n"
            "*.pyc\n"
            ".git/\n"
        )


def build_docker_image(
    dockerfile_path: Path,
    image_name: str,
    context_dir: Path,
    memory_limit: str = "8g",
    network_mode: str = "host",
) -> bool:
    """Build a Docker image from a Dockerfile with OOM protection."""
    # Create .dockerignore to speed up context transfer
    create_dockerignore(context_dir)
    
    # Use --memory to limit build memory and prevent OOM crashes
    # Use --network=host for faster package downloads (no NAT overhead)
    # Use --progress=plain for less output processing overhead
    cmd = [
        "docker", "build",
        "--memory", memory_limit,
        "--memory-swap", memory_limit,  # Disable swap to fail fast on OOM
        "--network", network_mode,  # Faster package downloads
        "--progress=plain",  # Less output overhead
        "-t", image_name,
        "-f", str(dockerfile_path),
        str(context_dir),
    ]
    print(f"Running: {' '.join(cmd)}")
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output_lines = []
        # Stream output to detect OOM early
        for line in iter(proc.stdout.readline, b''):
            decoded = line.decode('utf-8', errors='replace').rstrip()
            print(decoded)
            output_lines.append(decoded)
            # Check for OOM indicators in output
            if any(oom_indicator in decoded.lower() for oom_indicator in [
                'out of memory', 'oom', 'cannot allocate memory', 'killed',
                'memory allocation failed', 'enomem'
            ]):
                print(f"⚠️ Possible OOM detected during build for {image_name}")
        
        returncode = proc.wait(timeout=1800)  # 30 minute timeout for initial builds
        if returncode == 0:
            return True
        elif returncode == 137:  # SIGKILL - often indicates OOM
            print(f"Docker build killed (exit 137, likely OOM) for {image_name}")
            return False
        else:
            print(f"Docker build failed with return code {returncode}")
            return False
    except subprocess.TimeoutExpired:
        print(f"Docker build timed out for {image_name}, killing process...")
        if proc is not None:
            proc.kill()
            proc.wait()  # Ensure process is fully terminated
        return False
    except FileNotFoundError as e:
        print(f"Docker build error: {e}")
        return False


def run_initial_tests_docker(
    image_name: str,
    test_file_path: Path,
    memory_limit: str = "4g",
) -> tuple[bool, str]:
    """Run initial tests in a Docker container with OOM protection."""
    task_dir = test_file_path.parent
    test_filename = test_file_path.name
    
    # Mount the task directory to /mnt, copy test file to /home/user/, then run pytest
    # Use --memory and --memory-swap to prevent OOM crashes
    cmd = [
        "docker", "run", "--rm",
        "--memory", memory_limit,
        "--memory-swap", memory_limit,  # Disable swap to fail fast on OOM
        "--oom-kill-disable=false",  # Ensure OOM killer is active
        "-v", f"{task_dir}:/mnt",
        image_name,
        "bash", "-c",
        f"cp /mnt/{test_filename} /home/user/ && pytest -v /home/user/{test_filename}",
    ]
    print(f"Running: docker run --rm --memory {memory_limit} -v {task_dir}:/mnt {image_name} bash -c '...'")
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output_lines = []
        for line in iter(proc.stdout.readline, b''):
            decoded = line.decode('utf-8', errors='replace').rstrip()
            print(decoded)
            output_lines.append(decoded)
        
        returncode = proc.wait(timeout=300)
        output = '\n'.join(output_lines)
        
        if returncode == 0:
            return True, output
        elif returncode == 137:  # SIGKILL - often indicates OOM
            print(f"Test container killed (exit 137, likely OOM) for {image_name}")
            return False, f"Error running tests: OOM killed (exit 137)\n{output}"
        else:
            return False, output
    except subprocess.TimeoutExpired:
        print(f"Test execution timed out for {image_name}, killing process...")
        if proc is not None:
            proc.kill()
            proc.wait()  # Ensure process is fully terminated
        return False, "Error running tests: timeout"
    except FileNotFoundError as e:
        return False, f"Error running tests: {e}"


def get_failed_tasks_from_results(result_pattern: str = "retry*.json") -> List[Path]:
    """Read all result_*.json files in current directory and extract failed task directories."""
    current_dir = Path.cwd()
    result_files = glob.glob(str(current_dir / result_pattern))
    
    failed_task_dirs = set()
    
    for result_file in result_files:
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Extract failed tasks from results
        for result in data["results"]:
            if not result.get("success", True):  # Failed if success is False or missing
                task_dir_str = result.get("task_dir")
                task_dir = Path(task_dir_str)
                failed_task_dirs.add(task_dir)
    
    return list(sorted(list(failed_task_dirs)))


def process_task_directory(
    task_dir: Path,
    model: str = "gpt-5.1",
    provider: str = "openai",
    skip_build: bool = False,
    skip_tests: bool = False,
    reuse_dockerfile: bool = False,
    reuse_image: bool = False,
    build_memory_limit: str = "8g",
    test_memory_limit: str = "4g",
    network_mode: str = "host",
) -> Dict[str, Any]:
    """Process a single task directory: convert SIF to Dockerfile, build, and test."""
    result = {
        "task_dir": str(task_dir),
        "success": False,
        "error": None,
        "dockerfile_generated": False,
        "docker_build_success": False,
        "tests_passed": False,
    }
    
    def_path = task_dir / "container.def"
    initial_test_path = task_dir / "test_initial_state.py"
    dockerfile_path = task_dir / "Dockerfile"
    
    # Skip LLM conversion if Dockerfile exists and reuse is enabled
    if reuse_dockerfile and dockerfile_path.exists():
        print(f"Reusing existing Dockerfile for {task_dir.name}...")
        dockerfile_content = dockerfile_path.read_text(encoding="utf-8")
        result["dockerfile_generated"] = True
    else:
        # Read definition file (prefer .def, fallback to inspecting SIF)
        def_content = read_def_file(def_path)
        
        # Convert to Dockerfile using LLM
        print(f"Converting {task_dir.name} to Dockerfile using {provider}/{model}...")
        dockerfile_content = convert_def_to_dockerfile(
            def_content,
            model=model,
            provider=provider,
        )
        
        if not dockerfile_content:
            result["error"] = "Failed to generate Dockerfile"
            return result
        
        # Save Dockerfile
        dockerfile_path.write_text(dockerfile_content, encoding="utf-8")
        result["dockerfile_generated"] = True
    
    if skip_build:
        result["success"] = True
        return result
    
    # Build Docker image
    image_name = f"endless-task-{task_dir.name.lower().replace('_', '-')}"
    
    # Check if image already exists and reuse is enabled
    build_success = False
    if reuse_image:
        check_result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
        )
        if check_result.returncode == 0:
            print(f"Reusing existing Docker image {image_name}...")
            result["docker_build_success"] = True
            build_success = True
        else:
            print(f"Building Docker image {image_name}...")
            build_success = build_docker_image(
                dockerfile_path, image_name, task_dir,
                memory_limit=build_memory_limit,
                network_mode=network_mode,
            )
            result["docker_build_success"] = build_success
    else:
        print(f"Building Docker image {image_name}...")
        build_success = build_docker_image(
            dockerfile_path, image_name, task_dir,
            memory_limit=build_memory_limit,
            network_mode=network_mode,
        )
        result["docker_build_success"] = build_success
    
    if not build_success:
        result["error"] = "Docker build failed"
        return result
    
    if skip_tests:
        result["success"] = True
        return result
    
    # Run initial tests
    print(f"Running initial tests for {task_dir.name}...")
    tests_passed, test_output = run_initial_tests_docker(
        image_name, initial_test_path,
        memory_limit=test_memory_limit,
    )
    result["tests_passed"] = tests_passed
    
    if tests_passed:
        result["success"] = True
    else:
        result["error"] = f"Tests failed: {test_output[:500]}"
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert SIF files to Dockerfiles and verify initial tests pass"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("docker_conversion_results.json"),
        help="File to save the results to",
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("tasks"),
        help="Directory containing task directories",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model to use for conversion (e.g., Qwen/Qwen3-32B)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider to use (openai or anthropic)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip Docker build (only generate Dockerfile)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests (only build Docker image)",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=0,
        help="Start at task number",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=100,
        help="Number of tasks to process",
    )
    parser.add_argument(
        "--reuse-dockerfile",
        action="store_true",
        help="Reuse existing Dockerfile if present (skip LLM conversion)",
    )
    parser.add_argument(
        "--reuse-image",
        action="store_true",
        help="Reuse existing Docker image if present (skip build)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent workers to process tasks in parallel",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Read result_*.json files in current directory and retry only failed tasks",
    )
    parser.add_argument(
        "--pre-pull",
        action="store_true",
        help="Pre-pull base images before building (avoids redundant pulls during parallel builds)",
    )
    parser.add_argument(
        "--build-memory",
        type=str,
        default="8g",
        help="Memory limit for Docker build (e.g., 8g, 16g). Prevents OOM crashes.",
    )
    parser.add_argument(
        "--test-memory",
        type=str,
        default="4g",
        help="Memory limit for Docker test runs (e.g., 4g, 8g). Prevents OOM crashes.",
    )
    parser.add_argument(
        "--network",
        type=str,
        default="host",
        choices=["host", "bridge", "none"],
        help="Docker network mode for builds. 'host' is fastest for package downloads.",
    )
    args = parser.parse_args()
    
    
    if args.retry_failed:
        # Get failed tasks from result files
        task_dirs = get_failed_tasks_from_results()
        print(f"Found {len(task_dirs)} failed task directories from result files")
        task_dirs = task_dirs[args.start_at:args.start_at + args.num_tasks] 
        # replace path /data/v-kangandhi/endless with /home/v-kangandhi/
        task_dirs = [Path(str(task_dir).replace("/data/v-kangandhi/endless", "/home/v-kangandhi")) for task_dir in task_dirs]
    else:
        # Normal task selection logic
        task_dirs = [Path(args.task_dir) / f for f in os.listdir(args.task_dir) if "task" in f]
        # check if task dir has o3 summary
        task_dirs = [f for f in task_dirs if (f / "solutions" / "o3_summary.json").exists()]
        # check if o3 suummary pass @16 is greater than 0
        task_dirs = [f for f in task_dirs if json.load(open(f / "solutions" / "o3_summary.json"))["pass_at_k"]["16"] > 0]
        task_dirs = list(sorted(task_dirs))
        task_dirs = task_dirs[args.start_at:args.start_at + args.num_tasks]
        print(f"Found {len(task_dirs)} task directories")
    
    # Pre-pull base images if requested (avoids redundant pulls during parallel builds)
    if args.pre_pull and not args.skip_build:
        dockerfiles = [task_dir / "Dockerfile" for task_dir in task_dirs]
        pre_pull_base_images(dockerfiles)
    
    # Process each task directory
    results = []
    
    if args.workers <= 1:
        # Sequential processing
        for task_dir in tqdm(task_dirs, desc="Processing tasks"):
            result = process_task_directory(
                task_dir,
                model=args.model,
                provider=args.provider,
                skip_build=args.skip_build,
                skip_tests=args.skip_tests,
                reuse_dockerfile=args.reuse_dockerfile,
                reuse_image=args.reuse_image,
                build_memory_limit=args.build_memory,
                test_memory_limit=args.test_memory,
                network_mode=args.network,
            )
            results.append(result)
            
            if result["success"]:
                print(f"✅ {task_dir.name}: Success")
            else:
                print(f"❌ {task_dir.name}: {result.get('error', 'Unknown error')}")
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_task_directory,
                    task_dir,
                    model=args.model,
                    provider=args.provider,
                    skip_build=args.skip_build,
                    skip_tests=args.skip_tests,
                    reuse_dockerfile=args.reuse_dockerfile,
                    reuse_image=args.reuse_image,
                    build_memory_limit=args.build_memory,
                    test_memory_limit=args.test_memory,
                    network_mode=args.network,
                ): task_dir
                for task_dir in task_dirs
            }
            
            with tqdm(total=len(task_dirs), desc="Processing tasks") as pbar:
                for future in as_completed(futures):
                    task_dir = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result["success"]:
                            print(f"✅ {task_dir.name}: Success")
                        else:
                            print(f"❌ {task_dir.name}: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        print(f"❌ {task_dir.name}: Exception - {e}")
                        results.append({
                            "task_dir": str(task_dir),
                            "success": False,
                            "error": f"Exception: {e}",
                            "dockerfile_generated": False,
                            "docker_build_success": False,
                            "tests_passed": False,
                        })
                    finally:
                        pbar.update(1)
    
    # Calculate statistics
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total - successful
    
    # Count by stage
    dockerfile_generated = sum(1 for r in results if r.get("dockerfile_generated", False))
    docker_build_success = sum(1 for r in results if r.get("docker_build_success", False))
    tests_passed = sum(1 for r in results if r.get("tests_passed", False))
    
    # Count errors by stage
    errors_dockerfile = sum(1 for r in results if not r.get("dockerfile_generated", False))
    errors_build = sum(1 for r in results if r.get("dockerfile_generated", False) and not r.get("docker_build_success", False) and not args.skip_build)
    errors_tests = sum(1 for r in results if r.get("docker_build_success", False) and not r.get("tests_passed", False) and not args.skip_tests)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tasks: {total}")
    print(f"✅ Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
    
    print(f"\nStage Breakdown:")
    print(f"  Dockerfile generated: {dockerfile_generated}/{total} ({dockerfile_generated/total*100:.1f}%)")
    if not args.skip_build:
        print(f"  Docker build succeeded: {docker_build_success}/{total} ({docker_build_success/total*100:.1f}%)")
    if not args.skip_tests:
        print(f"  Tests passed: {tests_passed}/{total} ({tests_passed/total*100:.1f}%)")
    
    if failed > 0:
        print(f"\nErrors by Stage:")
        if errors_dockerfile > 0:
            print(f"  Dockerfile generation: {errors_dockerfile}")
        if errors_build > 0:
            print(f"  Docker build: {errors_build}")
        if errors_tests > 0:
            print(f"  Test execution: {errors_tests}")
    
    print("="*60)
    
    # Save results to JSON file
    output_file = Path(args.output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": total,
                "successful": successful,
                "failed": failed,
                "dockerfile_generated": dockerfile_generated,
                "docker_build_success": docker_build_success,
                "tests_passed": tests_passed,
                "errors_dockerfile": errors_dockerfile,
                "errors_build": errors_build,
                "errors_tests": errors_tests,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
